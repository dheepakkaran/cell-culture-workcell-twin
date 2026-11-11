"""Shared MuJoCo helpers: load the workcell, set robot configurations, check collisions,
attach a carried flask to the hand (kinematic weld)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene" / "workcell.xml"
FLASKS = ("flask_A", "flask_B", "flask_C")
ARM_DOF = 7
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.02  # flask body is 0.04 m thick -> each finger at 0.02
GRIP_OFFSET = 0.1034  # hand frame origin -> fingertip pad centre, along hand z
# Grasp orientation: hand z points down, fingers close along world y.
GRASP_ROT = np.diag([1.0, -1.0, -1.0])
PEN_TOL = 1e-3  # a contact counts as a collision only if penetration > 1 mm


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


class Workcell:
    """Wraps model + data. Robot-vs-world collision checking uses only geometry
    (mj_kinematics + mj_collision); a carried flask is moved with the hand."""

    def __init__(self, model: mujoco.MjModel | None = None):
        self.m = model if model is not None else load_model()
        self.d = mujoco.MjData(self.m)
        self.home = np.array([0.0, -0.6, 0.0, -2.6, 0.0, 2.0, 0.785])
        self.lo = self.m.jnt_range[:ARM_DOF, 0].copy()
        self.hi = self.m.jnt_range[:ARM_DOF, 1].copy()
        self.hand_id = self.m.body("hand").id
        robot_root = self.m.body("link0").id
        self.robot_bodies = {b for b in range(self.m.nbody) if self._is_desc(b, robot_root)}
        self.flask_body = {f: self.m.body(f).id for f in FLASKS}
        self.flask_qadr = {f: self.m.jnt_qposadr[self.m.body(f).jntadr[0]] for f in FLASKS}
        self.attached: str | None = None
        self.attach_rel = None  # (pos, rotmat) of flask in hand frame
        self.ignore_flask: str | None = None  # target flask during the grasp approach
        self.fingers = FINGER_OPEN
        self.defaults = {
            "rgba": self.m.geom_rgba.copy(),
            "contype": self.m.geom_contype.copy(),
            "conaff": self.m.geom_conaffinity.copy(),
            "light": self.m.light_diffuse.copy(),
        }
        self.reset()

    def _is_desc(self, b: int, root: int) -> bool:
        while b != 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ state
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.m.geom_rgba[:] = self.defaults["rgba"]
        self.m.geom_contype[:] = self.defaults["contype"]
        self.m.geom_conaffinity[:] = self.defaults["conaff"]
        self.m.light_diffuse[:] = self.defaults["light"]
        self.attached = None
        self.ignore_flask = None
        self.fingers = FINGER_OPEN
        self.set_q(self.home)

    def set_flask_pose(self, name: str, pos, yaw: float = 0.0):
        a = self.flask_qadr[name]
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def flask_pos(self, name: str) -> np.ndarray:
        a = self.flask_qadr[name]
        return self.d.qpos[a:a + 3].copy()

    def remove_cap(self, name: str):
        g = self.m.geom(f"{name}_cap").id
        self.m.geom_rgba[g, 3] = 0.0
        self.m.geom_contype[g] = 0
        self.m.geom_conaffinity[g] = 0

    def set_q(self, q):
        self.d.qpos[:ARM_DOF] = q
        self.d.qpos[7:9] = self.fingers
        mujoco.mj_kinematics(self.m, self.d)
        if self.attached is not None:
            hp = self.d.xpos[self.hand_id]
            hr = self.d.xmat[self.hand_id].reshape(3, 3)
            p = hp + hr @ self.attach_rel[0]
            r = hr @ self.attach_rel[1]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r.flatten())
            a = self.flask_qadr[self.attached]
            self.d.qpos[a:a + 3] = p
            self.d.qpos[a + 3:a + 7] = quat
            mujoco.mj_kinematics(self.m, self.d)

    def hand_pose(self, q=None):
        if q is not None:
            self.set_q(q)
        return self.d.xpos[self.hand_id].copy(), self.d.xmat[self.hand_id].reshape(3, 3).copy()

    def grip_point(self, q=None) -> np.ndarray:
        p, r = self.hand_pose(q)
        return p + r[:, 2] * GRIP_OFFSET

    def attach(self, name: str):
        hp, hr = self.hand_pose()
        a = self.flask_qadr[name]
        fr = np.zeros(9)
        mujoco.mju_quat2Mat(fr, self.d.qpos[a + 3:a + 7])
        fr = fr.reshape(3, 3)
        self.attach_rel = (hr.T @ (self.d.qpos[a:a + 3] - hp), hr.T @ fr)
        self.attached = name

    def detach(self):
        self.attached = None
        self.attach_rel = None

    # -------------------------------------------------------------- collision
    def _robotish(self, body: int) -> bool:
        if body in self.robot_bodies:
            return True
        return self.attached is not None and body == self.flask_body[self.attached]

    def contacts(self, q) -> list[tuple[str, str, float]]:
        """Penetrating contacts that involve the robot or the carried flask."""
        self.set_q(q)
        mujoco.mj_collision(self.m, self.d)
        out = []
        ign = self.flask_body[self.ignore_flask] if self.ignore_flask else None
        for c in self.d.contact[: self.d.ncon]:
            if c.dist > -PEN_TOL:
                continue
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            r1, r2 = self._robotish(b1), self._robotish(b2)
            if not (r1 or r2):
                continue
            if ign is not None and (b1 == ign or b2 == ign):
                continue
            if self.attached is not None:
                fb = self.flask_body[self.attached]
                # carried flask touching the hand/fingers holding it is expected
                if (b1 == fb and b2 in self.robot_bodies) or (b2 == fb and b1 in self.robot_bodies):
                    continue
            n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
            n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
            out.append((n1, n2, float(c.dist)))
        return out

    def in_collision(self, q) -> bool:
        if np.any(q < self.lo) or np.any(q > self.hi):
            return True
        return len(self.contacts(q)) > 0
