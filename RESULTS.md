# Results

Every number below is copied by `make_results.py` from `results/metrics.json`, written by
`eval.py --runs 100 --seed 0`. Per-run logs: `results/run_log.jsonl`. Simulation only.

## Setup
- Runs: **100**, 3 flasks per run (300 flask attempts).
- Randomization: flask xy noise uniform ±0.015 m, yaw ±5°, missing cap probability 0.1 per flask, hood light ×0.7–1.3.
- Sim cycle time: joints capped at 1.0 rad/s, gripper 0.5 s per action, settle 0.5 s.
- Collision rule: penetration > 1 mm between robot/carried flask and anything else.
- Execution: kinematic playback of planned joint paths; release + settle with MuJoCo physics.

## Outcomes
| Metric | Value |
|---|---|
| Run success rate (every capped flask placed, every capless flask rejected) | 100/100 = 1.000 |
| Flasks with cap | 258 |
| Flasks transferred (of those with cap) | 258 (1.000) |
| Flasks with missing cap | 42 |
| Missing caps rejected before grasp | 42 |
| Missing caps grasped anyway | 0 |
| Capped flasks falsely rejected | 0 |
| Failures by cause | none |

## Cycle time per transferred flask (sim seconds)
| mean | median | p95 | min | max |
|---|---|---|---|---|
| 7.48 | 7.35 | 9.26 | 5.42 | 11.73 |

Planning wall-clock time per transferred flask (s, this laptop, single process): mean 2.51, median 2.37, p95 3.84.

## Vision and placement accuracy
| Metric (m) | mean | median | p95 | max |
|---|---|---|---|---|
| Cap position error (vision vs sim ground truth) | 0.0027 | 0.0027 | 0.0034 | 0.0037 |
| Settled flask xy error vs shelf slot | 0.0002 | 0.0001 | 0.0005 | 0.0005 |

## RRT-Connect vs straight-line joint baseline (same start/goal per planning query)
Planning queries: 774; RRT-Connect solved 774.

| Planner | paths with a collision | colliding samples (0.01 rad spacing) |
|---|---|---|
| RRT-Connect + shortcut smoothing | 0 / 774 | 0 |
| Straight-line joint interpolation | 622 / 774 | 24681 |

| Segment | queries | RRT-Connect paths with collision | baseline paths with collision |
|---|---|---|---|
| home->pregrasp | 258 | 0 | 192 |
| pregrasp->preplace (carrying) | 258 | 0 | 258 |
| preplace->home | 258 | 0 | 172 |

Eval wall time: 764 s.
