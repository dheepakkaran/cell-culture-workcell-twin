import copy

import numpy as np
import pytest

from behaviors import TransferStateMachine, load_protocol, validate_protocol
from planner import count_collisions, ik_dls, rrt_connect, straight_line
from sim import GRASP_ROT, Workcell
from vision import OverheadCamera, detect_cap


@pytest.fixture(scope="module")
def wc():
    return Workcell()


@pytest.fixture(scope="module")
def cam(wc):
    c = OverheadCamera(wc)
    yield c
    c.close()


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def test_scene_has_hood_rack_shelf_flasks(wc):
    names = {wc.m.geom(i).name for i in range(wc.m.ngeom)}
    for g in ("hood_back", "hood_left", "hood_right", "hood_ceiling", "hood_sash", "rack_base",
              "incubator_body", "flask_A_cap", "flask_B_cap", "flask_C_cap"):
        assert g in names
    assert np.allclose(wc.m.geom("flask_A_body").size * 2, [0.08, 0.04, 0.15])  # T75-sized
    assert wc.m.camera("overhead").id >= 0


def test_ik_reaches_pose(wc):
    target = np.array([0.48, 0.32, 0.30])
    q, err = ik_dls(wc, target)
    assert q is not None
    assert np.linalg.norm(wc.grip_point(q) - target) < 2e-3
    _, rot = wc.hand_pose(q)
    assert np.linalg.norm(rot - GRASP_ROT) < 0.05
    wc.reset()


def test_path_collision_free(wc):
    """RRT-Connect between rack pre-grasp and shelf pre-place (over the divider):
    the smoothed path has zero colliding configurations when densely re-checked,
    while the straight-line joint baseline for the same query collides."""
    rng = np.random.default_rng(3)
    q_a, _ = ik_dls(wc, [0.36, 0.32, 0.30], rng=rng)
    q_b, _ = ik_dls(wc, [0.60, -0.32, 0.48], rng=rng)
    assert q_a is not None and q_b is not None
    path, info = rrt_connect(wc, q_a, q_b, rng=rng)
    assert path is not None, info
    assert np.allclose(path[0], q_a) and np.allclose(path[-1], q_b)
    assert count_collisions(wc, path, step=0.005) == 0
    assert count_collisions(wc, straight_line(wc, q_a, q_b)) > 0
    wc.reset()


def test_vision_finds_caps_within_tolerance(wc, cam, protocol):
    wc.reset()
    rgb, depth = cam.capture()
    for lw in protocol["labware"]:
        expected = np.array(protocol["slots"][lw["source_slot"]]) + [0, 0, 0.095]
        det = detect_cap(cam, rgb, depth, lw["cap_color"], expected)
        truth = wc.d.geom(f"{lw['id']}_cap").xpos + [0, 0, 0.01]
        assert det.present
        assert np.linalg.norm(det.world - truth) < 0.01  # 1 cm


def test_missing_cap_rejected(wc, cam, protocol):
    """A flask without a cap is rejected at verify_cap and never grasped or moved."""
    wc.reset()
    wc.remove_cap("flask_A")
    start = wc.flask_pos("flask_A")
    p = copy.deepcopy(protocol)
    p["labware"] = [lw for lw in p["labware"] if lw["id"] == "flask_A"]
    sm = TransferStateMachine(wc, cam, p, np.random.default_rng(0), compare_baseline=False)
    [res] = sm.run()
    assert res.outcome == "rejected"
    assert res.cause == "cap_missing"
    assert res.failed_step == "verify_cap"
    assert wc.attached is None
    assert np.allclose(wc.flask_pos("flask_A"), start)
    wc.reset()


def test_capped_flask_transferred(wc, cam, protocol):
    wc.reset()
    p = copy.deepcopy(protocol)
    p["labware"] = [lw for lw in p["labware"] if lw["id"] == "flask_B"]
    sm = TransferStateMachine(wc, cam, p, np.random.default_rng(1), compare_baseline=False)
    [res] = sm.run()
    assert res.outcome == "transferred", res.log
    dest = np.array(protocol["slots"]["incubator/2"])
    assert np.linalg.norm(wc.flask_pos("flask_B")[:2] - dest[:2]) < 0.02
    wc.reset()


def test_protocol_parses(protocol):
    assert protocol["steps"][0] == "locate"
    assert protocol["steps"].index("verify_cap") < protocol["steps"].index("grasp")
    assert protocol["on_fail"]["verify_cap"] == "skip_flask"
    assert len(protocol["labware"]) == 3
    bad = copy.deepcopy(protocol)
    bad["steps"] = ["grasp", "verify_cap"]
    with pytest.raises(ValueError):
        validate_protocol(bad)
