"""Motion planning for the workcell: damped-least-squares IK, RRT-Connect in joint space
with MuJoCo collision checking + shortcut smoothing, and a straight-line joint baseline."""
from __future__ import annotations

import mujoco
import numpy as np

from sim import ARM_DOF, GRASP_ROT, GRIP_OFFSET, Workcell


# ------------------------------------------------------------------------ IK
def ik_dls(wc: Workcell, target_grip, target_rot=GRASP_ROT, q0=None, iters=200,
           damping=0.05, tol_pos=1e-3, tol_rot=1e-2, rng=None, restarts=8):
    """DLS IK for the fingertip point + hand orientation. Returns (q, pos_err) or (None, err)."""
    rng = rng or np.random.default_rng(0)
    m, d = wc.m, wc.d
    target_hand = np.asarray(target_grip) - target_rot[:, 2] * GRIP_OFFSET
    best = (None, np.inf)
    seeds = [wc.home if q0 is None else np.asarray(q0)]
    for _ in range(restarts):
        seeds.append(rng.uniform(wc.lo, wc.hi) * 0.6 + wc.home * 0.4)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for seed in seeds:
        q = seed.copy()
        for _ in range(iters):
            wc.set_q(q)
            p = d.xpos[wc.hand_id]
            r = d.xmat[wc.hand_id].reshape(3, 3)
            e_p = target_hand - p
            e_r = 0.5 * (np.cross(r[:, 0], target_rot[:, 0]) + np.cross(r[:, 1], target_rot[:, 1])
                         + np.cross(r[:, 2], target_rot[:, 2]))
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                break
            mujoco.mj_comPos(m, d)  # Jacobian needs cdof
            mujoco.mj_jacBody(m, d, jacp, jacr, wc.hand_id)
            J = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), wc.lo, wc.hi)
        wc.set_q(q)
        err = np.linalg.norm(target_hand - d.xpos[wc.hand_id])
        r = d.xmat[wc.hand_id].reshape(3, 3)
        rot_err = np.linalg.norm(r - target_rot)
        if err < tol_pos * 2 and rot_err < 0.05 and not wc.in_collision(q):
            return q, err
        if err < best[1]:
            best = (None, err)
    return best


# ------------------------------------------------------------------ helpers
def edge_free(wc: Workcell, a, b, step=0.01) -> bool:
    n = max(2, int(np.ceil(np.max(np.abs(b - a)) / step)) + 1)
    for t in np.linspace(0, 1, n)[1:]:
        if wc.in_collision(a + t * (b - a)):
            return False
    return True


def densify(path, step=0.02):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


def count_collisions(wc: Workcell, path, step=0.01) -> int:
    """Number of densely sampled configurations along a path that are in collision."""
    return int(sum(wc.in_collision(q) for q in densify(path, step)))


def straight_line(wc: Workcell, q_start, q_goal):
    """Baseline: straight-line interpolation in joint space, no collision avoidance."""
    return np.array([q_start, q_goal])


# --------------------------------------------------------------- RRT-Connect
def rrt_connect(wc: Workcell, q_start, q_goal, rng=None, max_iters=4000, step=0.25,
                smooth_iters=150):
    """Bidirectional RRT (Kuffner & LaValle 2000) in the 7-D arm joint space.
    Returns (path ndarray, info dict) or (None, info)."""
    rng = rng or np.random.default_rng(0)
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    info = {"iters": 0, "nodes": 0}
    if wc.in_collision(q_start) or wc.in_collision(q_goal):
        info["error"] = "start_or_goal_in_collision"
        return None, info
    if edge_free(wc, q_start, q_goal):
        info["direct"] = True
        return np.array([q_start, q_goal]), info

    ta = {"q": [q_start], "p": [-1]}
    tb = {"q": [q_goal], "p": [-1]}

    def nearest(t, q):
        arr = np.asarray(t["q"])
        return int(np.argmin(np.sum((arr - q) ** 2, axis=1)))

    def extend(t, q):
        i = nearest(t, q)
        qn = t["q"][i]
        dlt = q - qn
        dist = np.linalg.norm(dlt)
        qnew = q if dist <= step else qn + dlt / dist * step
        if edge_free(wc, qn, qnew):
            t["q"].append(qnew)
            t["p"].append(i)
            return ("reached" if dist <= step else "advanced"), len(t["q"]) - 1
        return "trapped", -1

    def connect(t, q):
        while True:
            s, idx = extend(t, q)
            if s != "advanced":
                return s, idx

    def trace(t, i):
        out = []
        while i != -1:
            out.append(t["q"][i])
            i = t["p"][i]
        return out

    swapped = False
    for it in range(max_iters):
        info["iters"] = it + 1
        q_rand = q_goal if rng.random() < 0.05 else rng.uniform(wc.lo, wc.hi)
        s, ia = extend(ta, q_rand)
        if s != "trapped":
            s2, ib = connect(tb, ta["q"][ia])
            if s2 == "reached":
                pa = trace(ta, ia)[::-1]
                pb = trace(tb, ib)
                path = pa + pb[1:]
                if swapped:
                    path = path[::-1]
                info["nodes"] = len(ta["q"]) + len(tb["q"])
                info["raw_waypoints"] = len(path)
                path = shortcut(wc, np.array(path), rng, smooth_iters)
                info["smoothed_waypoints"] = len(path)
                return path, info
        ta, tb = tb, ta
        swapped = not swapped
    info["nodes"] = len(ta["q"]) + len(tb["q"])
    info["error"] = "max_iters"
    return None, info


def shortcut(wc: Workcell, path, rng, iters=150):
    """Random shortcut smoothing: replace sub-paths by straight collision-free edges."""
    path = [p for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_free(wc, path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return np.array(path)


def path_length(path) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


def path_duration(path, vmax=1.0) -> float:
    """Sim time for a path executed waypoint to waypoint with every joint capped at vmax rad/s
    (synchronised joints, constant speed per segment)."""
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.max(np.abs(np.diff(path, axis=0)), axis=1)) / vmax)


def cartesian_path(wc: Workcell, q_start, grip_goal, n=12, rng=None):
    """Straight-line fingertip motion (used for descend / lift / place / retreat):
    the grip point is interpolated in Cartesian space and each waypoint is solved by DLS IK,
    seeded from the previous one. Returns a joint path or None."""
    q_start = np.asarray(q_start, float)
    p0 = wc.grip_point(q_start)
    path = [q_start]
    for t in np.linspace(0, 1, n + 1)[1:]:
        q, _ = ik_dls(wc, p0 + t * (np.asarray(grip_goal) - p0), q0=path[-1], rng=rng, restarts=0)
        if q is None:
            return None
        path.append(q)
    return np.array(path)
