"""Replay evaluation run 0 (seed 0) with a camera on and save it as a GIF plus a frame strip.

Run 0 of `eval.py --runs 100 --seed 0` has flask_A and flask_C without caps and flask_B capped.
The same randomization and the same random generator are used, so the replay follows the logged
run exactly (checked against results/run_log.jsonl). Rendering does not change the simulation.
Output: docs/assets/protocol_run0.gif, docs/assets/protocol_run0_strip.png
"""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from behaviors import TransferStateMachine, load_protocol  # noqa: E402
from eval import randomize  # noqa: E402
from planner import densify  # noqa: E402
from sim import ROOT, Workcell  # noqa: E402
from vision import OverheadCamera  # noqa: E402

W, H = 480, 360
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


class Recorder(TransferStateMachine):
    """State machine that renders the arm while it plays back each planned path."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # a separate copy of the model for the picture only: hood ceiling and front glass hidden,
        # so the simulated model, its collisions and the vision camera stay untouched
        self.vm = mujoco.MjModel.from_xml_path(str(ROOT / "scene" / "workcell.xml"))
        self.hidden = [mujoco.mj_name2id(self.vm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("hood_ceiling", "hood_sash")]
        self.vd = mujoco.MjData(self.vm)
        self.r = mujoco.Renderer(self.vm, H, W)
        self.fc = mujoco.MjvCamera()
        self.fc.lookat[:] = [0.42, 0.0, 0.12]
        self.fc.distance, self.fc.azimuth, self.fc.elevation = 1.75, -25, -55
        self.frames, self.keys, self.caption = [], [], ""

    def snap(self, key=None):
        self.vm.geom_rgba[:] = self.wc.m.geom_rgba          # removed caps stay invisible
        self.vm.geom_rgba[self.hidden, 3] = 0
        self.vm.light_diffuse[:] = self.wc.m.light_diffuse
        self.vd.qpos[:] = self.wc.d.qpos
        mujoco.mj_kinematics(self.vm, self.vd)
        self.r.update_scene(self.vd, self.fc)
        img = np.ascontiguousarray(self.r.render())
        cv2.rectangle(img, (0, 0), (W, 24), (255, 255, 255), -1)
        cv2.putText(img, self.caption, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
        self.frames.append(img)
        if key:
            self.keys.append((key, img))

    def _execute(self, r, path, label, ignore=None):
        super()._execute(r, path, label, ignore)       # same checks and timing as the evaluation
        for q in densify(path, 0.06)[::1]:
            self.wc.set_q(q)
            self.snap()
        self.wc.set_q(self.q)

    def run_flask(self, lw):
        self.caption = f"{lw['id']}: locate + verify cap"
        self.snap()
        res, action = super().run_flask(lw)
        verdict = {"transferred": "placed on incubator shelf", "rejected": f"REJECTED ({res.cause}), not touched"}
        self.caption = f"{lw['id']}: {verdict.get(res.outcome, res.outcome)}"
        for _ in range(10):
            self.snap()
        self.keys.append((self.caption, self.frames[-1]))
        return res, action

    def step_grasp(self, r, lw, ctx):
        super().step_grasp(r, lw, ctx)
        self.caption = f"{lw['id']}: grasped, RRT-Connect path over the divider"
        self.snap(self.caption)

    def step_place(self, r, lw, ctx):
        self.caption = f"{lw['id']}: place on shelf"
        super().step_place(r, lw, ctx)
        self.snap(self.caption)


def main():
    protocol = load_protocol()
    wc = Workcell()
    cam = OverheadCamera(wc)
    rng = np.random.default_rng(0)
    wc.reset()
    truth = randomize(wc, protocol, rng, 0.015, 0.10)
    sm = Recorder(wc, cam, protocol, rng)
    sm.caption = "run 0 (seed 0): " + ", ".join(f"{k} {'cap' if v else 'NO cap'}" for k, v in truth.items())
    for _ in range(12):
        sm.snap()
    sm.keys.append((sm.caption, sm.frames[-1]))
    results = sm.run()
    cam.close()

    logged = json.loads(open(ROOT / "results" / "run_log.jsonl").readline())
    for fr, lg in zip(results, logged["flasks"]):
        assert fr.outcome == lg["outcome"] and abs(fr.sim_time_s - lg["sim_time_s"]) < 1e-9, (fr, lg)
    print("replay matches run_log.jsonl run 0:", [(f.flask, f.outcome, round(f.sim_time_s, 2)) for f in results])

    frames = [Image.fromarray(f).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in sm.frames]
    frames[0].save(OUT / "protocol_run0.gif", save_all=True, append_images=frames[1:], duration=80, loop=0,
                   optimize=True)
    picks = [img for _, img in sm.keys][:6]
    strip = np.concatenate([np.concatenate(picks[i:i + 3], axis=1) for i in (0, 3)], axis=0)
    Image.fromarray(strip).save(OUT / "protocol_run0_strip.png", optimize=True)
    print(len(sm.frames), "frames;", [k for k, _ in sm.keys])


if __name__ == "__main__":
    main()
