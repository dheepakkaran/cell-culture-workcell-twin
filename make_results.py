"""Write RESULTS.md from results/metrics.json (no number is typed by hand)."""
import json

from sim import ROOT

m = json.load(open(ROOT / "results" / "metrics.json"))
pq = m["planning_queries"]
c = m["cycle_time_s_per_flask"]
pw = m["planning_wall_time_s_per_flask"]
ve = m["vision_cap_position_error_m"]
pe = m["place_xy_error_m"]
cfg = m["config"]


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


lines = [
    "# Results",
    "",
    "Every number below is copied by `make_results.py` from `results/metrics.json`, written by",
    f"`eval.py --runs {cfg['n_runs']} --seed {cfg['seed']}`. Per-run logs: `results/run_log.jsonl`. Simulation only.",
    "",
    "## Setup",
    f"- Runs: **{m['n_runs']}**, 3 flasks per run ({m['flasks_total']} flask attempts).",
    f"- Randomization: flask xy noise uniform ±{cfg['pose_noise_m']} m, yaw ±5°, missing cap probability {cfg['missing_cap_p']} per flask, hood light ×0.7–1.3.",
    f"- Sim cycle time: joints capped at {cfg['joint_speed_rad_s']} rad/s, gripper {cfg['gripper_time_s']} s per action, settle {cfg['settle_time_s']} s.",
    f"- Collision rule: {cfg['collision_rule']}.",
    f"- Execution: {cfg['execution']}.",
    "",
    "## Outcomes",
    "| Metric | Value |",
    "|---|---|",
    f"| Run success rate (every capped flask placed, every capless flask rejected) | {m['runs_successful']}/{m['n_runs']} = {m['run_success_rate']:.3f} |",
    f"| Flasks with cap | {m['flasks_with_cap']} |",
    f"| Flasks transferred (of those with cap) | {m['flasks_transferred']} ({m['flask_transfer_rate_when_cap_present']:.3f}) |",
    f"| Flasks with missing cap | {m['flasks_missing_cap']} |",
    f"| Missing caps rejected before grasp | {m['missing_caps_rejected']} |",
    f"| Missing caps grasped anyway | {m['missing_caps_grasped']} |",
    f"| Capped flasks falsely rejected | {m['cap_present_false_rejects']} |",
    f"| Failures by cause | {json.dumps(m['failures_by_cause']) if m['failures_by_cause'] else 'none'} |",
    "",
    "## Cycle time per transferred flask (sim seconds)",
    "| mean | median | p95 | min | max |",
    "|---|---|---|---|---|",
    f"| {f(c['mean'], 2)} | {f(c['median'], 2)} | {f(c['p95'], 2)} | {f(c['min'], 2)} | {f(c['max'], 2)} |" if c else "| n/a | | | | |",
    "",
    "Planning wall-clock time per transferred flask (s, this laptop, single process): "
    + (f"mean {f(pw['mean'], 2)}, median {f(pw['median'], 2)}, p95 {f(pw['p95'], 2)}." if pw else "n/a"),
    "",
    "## Vision and placement accuracy",
    "| Metric (m) | mean | median | p95 | max |",
    "|---|---|---|---|---|",
    f"| Cap position error (vision vs sim ground truth) | {f(ve['mean'], 4)} | {f(ve['median'], 4)} | {f(ve['p95'], 4)} | {f(ve['max'], 4)} |" if ve else "| cap | n/a | | | |",
    f"| Settled flask xy error vs shelf slot | {f(pe['mean'], 4)} | {f(pe['median'], 4)} | {f(pe['p95'], 4)} | {f(pe['max'], 4)} |" if pe else "| place | n/a | | | |",
    "",
    "## RRT-Connect vs straight-line joint baseline (same start/goal per planning query)",
    f"Planning queries: {pq['total']}; RRT-Connect solved {pq['rrt_connect_solved']}.",
    "",
    "| Planner | paths with a collision | colliding samples (0.01 rad spacing) |",
    "|---|---|---|",
    f"| RRT-Connect + shortcut smoothing | {pq['rrt_connect_paths_with_collision']} / {pq['rrt_connect_solved']} | {pq['rrt_connect_colliding_samples']} |",
    f"| Straight-line joint interpolation | {pq['baseline_paths_with_collision']} / {pq['total']} | {pq['baseline_colliding_samples']} |",
    "",
    "| Segment | queries | RRT-Connect paths with collision | baseline paths with collision |",
    "|---|---|---|---|",
]
for seg, s in pq["by_segment"].items():
    lines.append(f"| {seg} | {s['queries']} | {s['rrt_paths_with_collision']} | {s['baseline_paths_with_collision']} |")
lines += ["", f"Eval wall time: {m['eval_wall_time_s']:.0f} s.", ""]
(ROOT / "RESULTS.md").write_text("\n".join(lines))
print("wrote RESULTS.md")
