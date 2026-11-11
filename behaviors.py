"""Robot behaviors built from the YAML lab protocol.

Each flask runs the protocol's step list as a state machine:
locate -> verify_cap -> approach -> grasp -> move -> place -> release -> retreat.
A failed step follows the protocol's on_fail action (skip_flask or abort_run).
Execution is kinematic along planned joint paths (every configuration collision-checked);
release hands the flask to MuJoCo physics, which settles it on the shelf."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from planner import (cartesian_path, count_collisions, densify, ik_dls, path_duration, rrt_connect,
                     straight_line)
from sim import FINGER_CLOSED, FINGER_OPEN, ROOT, Workcell
from vision import OverheadCamera, detect_cap

KNOWN_STEPS = ["locate", "verify_cap", "approach", "grasp", "move", "place", "release", "retreat"]
KNOWN_ACTIONS = {"skip_flask", "abort_run"}


def load_protocol(path=ROOT / "protocol.yaml") -> dict:
    with open(path) as fh:
        p = yaml.safe_load(fh)
    validate_protocol(p)
    return p


def validate_protocol(p: dict):
    for key in ("protocol", "labware", "slots", "steps", "on_fail", "params"):
        if key not in p:
            raise ValueError(f"protocol missing '{key}'")
    unknown = [s for s in p["steps"] if s not in KNOWN_STEPS]
    if unknown:
        raise ValueError(f"unknown steps: {unknown}")
    order = [KNOWN_STEPS.index(s) for s in p["steps"]]
    if order != sorted(order):
        raise ValueError("steps out of order")
    if "verify_cap" in p["steps"] and p["steps"].index("verify_cap") > p["steps"].index("grasp"):
        raise ValueError("verify_cap must come before grasp")
    for step, action in p["on_fail"].items():
        if action not in KNOWN_ACTIONS:
            raise ValueError(f"unknown on_fail action {action} for {step}")
    for lw in p["labware"]:
        for k in ("id", "cap_color", "source_slot", "dest_slot"):
            if k not in lw:
                raise ValueError(f"labware entry missing '{k}'")
        if lw["source_slot"] not in p["slots"] or lw["dest_slot"] not in p["slots"]:
            raise ValueError(f"unknown slot for {lw['id']}")


class StepFailed(Exception):
    def __init__(self, cause: str, detail: str = ""):
        super().__init__(f"{cause}: {detail}")
        self.cause, self.detail = cause, detail


@dataclass
class FlaskResult:
    flask: str
    outcome: str = "pending"        # transferred | rejected | failed
    cause: str = ""                 # failure / rejection cause
    failed_step: str = ""
    cap_present_truth: bool = True
    sim_time_s: float = 0.0
    plan_wall_s: float = 0.0
    vision_error_m: float | None = None
    place_error_m: float | None = None
    queries: list = field(default_factory=list)  # RRT vs straight-line per planning query
    log: list = field(default_factory=list)


class TransferStateMachine:
    def __init__(self, wc: Workcell, cam: OverheadCamera, protocol: dict, rng: np.random.Generator,
                 compare_baseline: bool = True):
        self.wc, self.cam, self.p, self.rng = wc, cam, protocol, rng
        self.par = protocol["params"]
        self.slots = {k: np.array(v, float) for k, v in protocol["slots"].items()}
        self.compare_baseline = compare_baseline
        self.q = wc.home.copy()

    # ---------------------------------------------------------------- helpers
    def _ik(self, grip, ignore=None):
        self.wc.ignore_flask = ignore
        q, err = ik_dls(self.wc, grip, q0=self.q, rng=self.rng)
        self.wc.ignore_flask = None
        return q

    def _plan(self, r: FlaskResult, q_goal, label):
        t0 = time.perf_counter()
        path, info = rrt_connect(self.wc, self.q, q_goal, rng=self.rng)
        r.plan_wall_s += time.perf_counter() - t0
        q_entry = {"segment": label, "rrt_ok": path is not None,
                   "rrt_nodes": info.get("nodes", 0), "rrt_waypoints": info.get("smoothed_waypoints", 2 if path is not None else 0)}
        if self.compare_baseline:
            q_entry["baseline_colliding_samples"] = count_collisions(self.wc, straight_line(self.wc, self.q, q_goal))
        if path is None:
            r.queries.append(q_entry)
            raise StepFailed("plan_failed", f"{label}: {info.get('error')}")
        q_entry["rrt_colliding_samples"] = count_collisions(self.wc, path)
        r.queries.append(q_entry)
        return path

    def _execute(self, r: FlaskResult, path, label, ignore=None):
        """Play the path back (dense, 0.01 rad), collision-check every configuration."""
        self.wc.ignore_flask = ignore
        hits = [q for q in densify(path, 0.01) if self.wc.in_collision(q)]
        self.wc.ignore_flask = None
        if hits:
            raise StepFailed("execution_collision", f"{label}: {len(hits)} configs in collision")
        r.sim_time_s += path_duration(path, self.par["joint_speed_rad_s"])
        self.q = np.array(path[-1])
        self.wc.set_q(self.q)

    def _cartesian(self, r, grip_goal, label, ignore=None):
        self.wc.ignore_flask = ignore
        path = cartesian_path(self.wc, self.q, grip_goal, rng=self.rng)
        self.wc.ignore_flask = None
        if path is None:
            raise StepFailed("ik_failed", f"{label}: no IK along straight fingertip line")
        self._execute(r, path, label, ignore=ignore)

    def _log(self, r, step, msg):
        r.log.append(f"{step}: {msg}")

    # ------------------------------------------------------------------ steps
    def step_locate(self, r, lw, ctx):
        rgb, depth = self.cam.capture()
        expected_cap = self.slots[lw["source_slot"]] + [0, 0, 0.095]  # cap top centre
        ctx["det"] = detect_cap(self.cam, rgb, depth, lw["cap_color"], expected_cap)
        ctx["expected_cap"] = expected_cap
        self._log(r, "locate", f"present={ctx['det'].present} area={ctx['det'].area_px}px")

    def step_verify_cap(self, r, lw, ctx):
        det = ctx["det"]
        if not det.present:
            raise StepFailed("cap_missing", det.reason)
        off = np.linalg.norm(det.world[:2] - ctx["expected_cap"][:2])
        if off > self.par["position_tolerance_m"]:
            raise StepFailed("pose_out_of_tolerance", f"{off:.3f} m")
        cap_gt = self.wc.d.geom(f"{lw['id']}_cap").xpos + [0, 0, 0.01]
        r.vision_error_m = float(np.linalg.norm(det.world - cap_gt))
        self._log(r, "verify_cap", f"cap verified at {np.round(det.world, 3).tolist()}")

    def step_approach(self, r, lw, ctx):
        cap = ctx["det"].world
        grip = cap - [0, 0, 0.02 + self.par["grasp_height_above_body_top_m"]]
        pre = np.array([grip[0], grip[1], self.par["pregrasp_height_m"]])
        q_pre = self._ik(pre)
        q_grasp = self._ik(grip, ignore=lw["id"])
        if q_pre is None or q_grasp is None:
            raise StepFailed("ik_failed", "pre-grasp" if q_pre is None else "grasp")
        path = self._plan(r, q_pre, "home->pregrasp")
        self._execute(r, path, "home->pregrasp")
        self._cartesian(r, grip, "descend", ignore=lw["id"])
        ctx["pre"] = pre

    def step_grasp(self, r, lw, ctx):
        self.wc.fingers = FINGER_CLOSED
        self.wc.set_q(self.q)
        self.wc.attach(lw["id"])
        r.sim_time_s += self.par["gripper_time_s"]
        ctx["grip_offset"] = self.wc.grip_point(self.q) - self.wc.flask_pos(lw["id"])
        self._log(r, "grasp", "gripper closed, flask welded to hand")

    def step_move(self, r, lw, ctx):
        self._cartesian(r, ctx["pre"], "lift")
        dest = self.slots[lw["dest_slot"]]
        place_grip = dest + ctx["grip_offset"]
        pre_place = np.array([place_grip[0], place_grip[1], self.par["carry_height_m"]])
        q_pp = self._ik(pre_place)
        q_place = self._ik(place_grip, ignore=lw["id"])
        if q_pp is None or q_place is None:
            raise StepFailed("ik_failed", "pre-place" if q_pp is None else "place")
        path = self._plan(r, q_pp, "pregrasp->preplace (carrying)")
        self._execute(r, path, "pregrasp->preplace")
        ctx["pre_place"], ctx["place_grip"] = pre_place, place_grip

    def step_place(self, r, lw, ctx):
        self._cartesian(r, ctx["place_grip"], "place descend")

    def step_release(self, r, lw, ctx):
        wc = self.wc
        wc.detach()
        wc.fingers = FINGER_OPEN
        wc.set_q(self.q)
        r.sim_time_s += self.par["gripper_time_s"]
        # hand the flask to physics: hold the arm with its position actuators and settle
        wc.d.qvel[:] = 0
        wc.d.ctrl[:7] = self.q
        wc.d.ctrl[7] = 255
        mujoco.mj_forward(wc.m, wc.d)
        for _ in range(int(self.par["settle_time_s"] / wc.m.opt.timestep)):
            mujoco.mj_step(wc.m, wc.d)
        wc.d.qvel[:] = 0
        wc.set_q(self.q)
        r.sim_time_s += self.par["settle_time_s"]
        pos = wc.flask_pos(lw["id"])
        a = wc.flask_qadr[lw["id"]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, wc.d.qpos[a + 3:a + 7])
        tilt = np.degrees(np.arccos(np.clip(rot.reshape(3, 3)[2, 2], -1, 1)))
        err = float(np.linalg.norm(pos[:2] - self.slots[lw["dest_slot"]][:2]))
        r.place_error_m = err
        if err > self.par["place_tolerance_m"] or tilt > 10 or pos[2] < 0.25:
            raise StepFailed("placement_error", f"xy err {err:.3f} m, tilt {tilt:.1f} deg")
        self._log(r, "release", f"settled, xy err {err*1000:.1f} mm, tilt {tilt:.1f} deg")

    def step_retreat(self, r, lw, ctx):
        self._cartesian(r, ctx["pre_place"], "retreat up")
        path = self._plan(r, self.wc.home, "preplace->home")
        self._execute(r, path, "preplace->home")

    # ------------------------------------------------------------------- run
    def run_flask(self, lw) -> tuple[FlaskResult, str | None]:
        r = FlaskResult(lw["id"])
        ctx: dict = {}
        for step in self.p["steps"]:
            try:
                getattr(self, f"step_{step}")(r, lw, ctx)
            except StepFailed as e:
                action = self.p["on_fail"].get(step, "abort_run")
                r.failed_step, r.cause = step, e.cause
                r.outcome = "rejected" if step in ("locate", "verify_cap") else "failed"
                self._log(r, step, f"FAILED {e} -> {action}")
                self._recover(lw)
                return r, action
        r.outcome = "transferred"
        return r, None

    def _recover(self, lw):
        """After a failure: drop any weld and reset the arm to home for the next flask."""
        if self.wc.attached:
            self.wc.detach()
        self.wc.fingers = FINGER_OPEN
        self.q = self.wc.home.copy()
        self.wc.set_q(self.q)

    def run(self) -> list[FlaskResult]:
        results = []
        for lw in self.p["labware"]:
            r, action = self.run_flask(lw)
            results.append(r)
            if action == "abort_run":
                for rest in self.p["labware"][len(results):]:
                    results.append(FlaskResult(rest["id"], outcome="not_attempted", cause="run_aborted"))
                break
        return results
