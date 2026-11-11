"""Randomized evaluation: N runs of the protocol with flask pose noise, missing caps and
lighting changes. Writes results/metrics.json and results/run_log.jsonl."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from behaviors import TransferStateMachine, load_protocol
from sim import ROOT, Workcell
from vision import OverheadCamera

RESULTS = ROOT / "results"


def randomize(wc: Workcell, protocol, rng, pose_noise, missing_cap_p):
    truth = {}
    for lw in protocol["labware"]:
        nominal = np.array(protocol["slots"][lw["source_slot"]], float)
        pos = nominal + [rng.uniform(-pose_noise, pose_noise), rng.uniform(-pose_noise, pose_noise), 0]
        wc.set_flask_pose(lw["id"], pos, yaw=np.deg2rad(rng.uniform(-5, 5)))
        has_cap = rng.random() >= missing_cap_p
        if not has_cap:
            wc.remove_cap(lw["id"])
        truth[lw["id"]] = has_cap
    wc.m.light_diffuse[:] = wc.defaults["light"] * rng.uniform(0.7, 1.3)
    wc.set_q(wc.home)
    return truth


def run_eval(n_runs=100, seed=0, pose_noise=0.015, missing_cap_p=0.10, log_path=None, verbose=True):
    protocol = load_protocol()
    wc = Workcell()
    cam = OverheadCamera(wc)
    runs = []
    t_start = time.perf_counter()
    log_fh = open(log_path, "w") if log_path else None
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)
        wc.reset()
        truth = randomize(wc, protocol, rng, pose_noise, missing_cap_p)
        sm = TransferStateMachine(wc, cam, protocol, rng)
        flasks = sm.run()
        for fr in flasks:
            fr.cap_present_truth = truth[fr.flask]
        ok = all((fr.outcome == "transferred") if truth[fr.flask] else
                 (fr.outcome == "rejected" and fr.cause == "cap_missing") for fr in flasks)
        rec = {"run": i, "seed": seed + i, "success": ok,
               "flasks": [fr.__dict__ for fr in flasks]}
        runs.append(rec)
        if log_fh:
            log_fh.write(json.dumps(rec, default=float) + "\n")
            log_fh.flush()
        if verbose:
            summ = " ".join(f"{fr.flask}:{fr.outcome}{'(' + fr.cause + ')' if fr.cause else ''}" for fr in flasks)
            print(f"run {i:3d} success={ok} {summ} [{time.perf_counter() - t_start:.0f}s]", flush=True)
    cam.close()
    if log_fh:
        log_fh.close()
    return runs, time.perf_counter() - t_start, dict(seed=seed, pose_noise_m=pose_noise,
                                                     missing_cap_p=missing_cap_p)


def summarize(runs, wall_s, cfg):
    flasks = [f for r in runs for f in r["flasks"]]
    transferred = [f for f in flasks if f["outcome"] == "transferred"]
    cyc = np.array([f["sim_time_s"] for f in transferred])
    plan_wall = np.array([f["plan_wall_s"] for f in transferred])
    queries = [q for f in flasks for q in f["queries"]]
    cap_missing = [f for f in flasks if not f["cap_present_truth"]]
    cap_present = [f for f in flasks if f["cap_present_truth"]]
    failures = Counter(f["cause"] for f in flasks
                       if f["outcome"] not in ("transferred",) and not (f["outcome"] == "rejected"
                                                                       and not f["cap_present_truth"]
                                                                       and f["cause"] == "cap_missing"))
    verr = np.array([f["vision_error_m"] for f in flasks if f["vision_error_m"] is not None])
    perr = np.array([f["place_error_m"] for f in transferred if f["place_error_m"] is not None])
    rrt_q = [q for q in queries if q["rrt_ok"]]

    def stats(a):
        if len(a) == 0:
            return None
        return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "config": {**cfg, "n_runs": len(runs), "flasks_per_run": 3,
                   "joint_speed_rad_s": 1.0, "gripper_time_s": 0.5, "settle_time_s": 0.5,
                   "collision_rule": "penetration > 1 mm between robot/carried flask and anything else",
                   "execution": "kinematic playback of planned joint paths; release + settle with MuJoCo physics"},
        "n_runs": len(runs),
        "run_success_rate": float(np.mean([r["success"] for r in runs])),
        "runs_successful": int(sum(r["success"] for r in runs)),
        "flasks_total": len(flasks),
        "flasks_with_cap": len(cap_present),
        "flasks_missing_cap": len(cap_missing),
        "flasks_transferred": len(transferred),
        "flask_transfer_rate_when_cap_present": len(transferred) / max(1, len(cap_present)),
        "missing_caps_rejected": sum(f["outcome"] == "rejected" and f["cause"] == "cap_missing" for f in cap_missing),
        "missing_caps_grasped": sum(f["outcome"] not in ("rejected", "not_attempted") for f in cap_missing),
        "cap_present_false_rejects": sum(f["outcome"] == "rejected" for f in cap_present),
        "failures_by_cause": dict(failures),
        "cycle_time_s_per_flask": stats(cyc),
        "planning_wall_time_s_per_flask": stats(plan_wall),
        "vision_cap_position_error_m": stats(verr),
        "place_xy_error_m": stats(perr),
        "planning_queries": {
            "total": len(queries),
            "rrt_connect_solved": len(rrt_q),
            "rrt_connect_paths_with_collision": sum(q["rrt_colliding_samples"] > 0 for q in rrt_q),
            "rrt_connect_colliding_samples": int(sum(q["rrt_colliding_samples"] for q in rrt_q)),
            "baseline_paths_with_collision": sum(q["baseline_colliding_samples"] > 0 for q in queries),
            "baseline_colliding_samples": int(sum(q["baseline_colliding_samples"] for q in queries)),
            "by_segment": {
                seg: {"queries": sum(q["segment"] == seg for q in queries),
                      "rrt_paths_with_collision": sum(q["segment"] == seg and q.get("rrt_colliding_samples", 0) > 0 for q in queries),
                      "baseline_paths_with_collision": sum(q["segment"] == seg and q["baseline_colliding_samples"] > 0 for q in queries)}
                for seg in sorted({q["segment"] for q in queries})},
        },
        "eval_wall_time_s": wall_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "metrics.json"))
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, wall, cfg = run_eval(a.runs, a.seed, log_path=RESULTS / "run_log.jsonl")
    m = summarize(runs, wall, cfg)
    with open(a.out, "w") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps({k: m[k] for k in ("n_runs", "run_success_rate", "failures_by_cause")}, indent=2))


if __name__ == "__main__":
    main()
