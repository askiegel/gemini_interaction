"""Offline stopped camera checks; visible Marvin is not collision-object proof."""
import time

import pytest

from marvin_blocked_wait import blocked_wait_diagnostics
from test_find_marvin_closed_loop import make_runtime, motions
from test_marvin_continuous_lidar_recovery import certificate, prepare
from test_marvin_vision_collision_only import offline  # Enforce no network/process/ranging.


def boundary_case(tmp_path, monkeypatch, *, point=(.3, 0.), visual=(0, .8),
                  clear_after=None, frozen=False, fault=None):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8), visual], calibrated_arrival=False)
    r, b, robot, events, clock = bundle
    flags = prepare(bundle, monkeypatch, clear_after=clear_after, frozen=frozen, fault=fault)
    original = r.world_model.get_lidar_obstacles
    def lidar(**kwargs):
        snapshot = original(**kwargs)
        for p in snapshot['local_motion_geometry']['points']:
            if p['x_m'] == .3 and p['y_m'] == 0.:
                p.update(x_m=point[0], y_m=point[1])
        return snapshot
    r.world_model.get_lidar_obstacles = lidar
    observation = r._observe_find_marvin_v2()
    floor = observation['source_frame_stamp_ns']
    _, cert = certificate()
    entry = lidar()
    flags.update(waiting=True, began=time.monotonic())
    d, history = blocked_wait_diagnostics(), []
    def execute(guard=lambda: True):
        return r._wait_for_marvin_blocked_route(expected_session=r.lidar_worker.session,
            lidar=entry, association=observation['arrival'], execution_guard=guard,
            diagnostics=d, history=history, interruption=cert, camera_source_floor=floor)
    return bundle, flags, d, history, floor, execute


@pytest.mark.parametrize('scene', ['marvin_ahead', 'foreground_object_with_marvin_behind'])
def test_indistinguishable_front_boundary_scenes_cannot_manufacture_completion(tmp_path, monkeypatch, scene):
    # These physically different causes expose identical current sensor fields.
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch)
    r, b, robot, events, clock = bundle
    result = execute()
    record, = d['collision_boundary_observations']
    assert not result['ok'] and result['reason'] == 'find_marvin_blocked_wait_exhausted'
    assert record['camera_recheck_performed'] and record['marvin_confirmed_forward']
    assert record['reason'] == 'marvin_collision_boundary_association_unverified'
    assert not record['boundary_object_identity_verified']
    assert not record['safe_approach_complete'] and not record['arrived_at_marvin']
    assert not record['motion_authorized'] and not motions(events)
    assert events.index('stop') < len(events)-1 and events[-1] == 'observe'
    assert result['camera_floor'] > floor and len(b.stamps) == 2
    assert r._marvin_alignment_observation is None
    assert d['blocked_wait_recheck_count'] == 12 and d['blocked_wait_total_seconds'] <= 30


@pytest.mark.parametrize('point', [(0., .3), (0., -.3), (-.3, 0.), (.45, 0.)])
def test_side_rear_and_inclusive_boundary_never_become_target_arrival(tmp_path, monkeypatch, point):
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch, point=point)
    assert not execute()['ok']
    record, = d['collision_boundary_observations']
    assert record['camera_recheck_performed']
    assert not record['boundary_object_identity_verified'] and not record['safe_approach_complete']
    assert not motions(bundle[3])


@pytest.mark.parametrize('fault', ['lost', 'off_axis', 'duplicate_camera', 'stale_camera'])
def test_boundary_camera_must_be_independent_fresh_and_forward(tmp_path, monkeypatch, fault):
    visual = 'lost' if fault == 'lost' else (120, .8) if fault == 'off_axis' else (0, .8)
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch, visual=visual)
    b = bundle[1]
    b.repeat_camera = fault == 'duplicate_camera'
    b.stale_camera = fault == 'stale_camera'
    assert not execute()['ok']
    record, = d['collision_boundary_observations']
    assert record['camera_recheck_performed'] and not record['marvin_confirmed_forward']
    assert not record['safe_approach_complete'] and not motions(bundle[3])
    assert bundle[0]._marvin_alignment_observation is None


@pytest.mark.parametrize('fault', ['session', 'regression', 'stale', 'coverage', 'owner',
                                  'quarantine', 'nonzero', 'controller_stale', 'frozen'])
def test_invalid_sensing_or_controller_cannot_supply_boundary_evidence(tmp_path, monkeypatch, fault):
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch,
        frozen=fault == 'frozen', fault=None if fault == 'frozen' else fault)
    assert not execute()['ok']
    assert not d.get('collision_boundary_observations')
    assert len(bundle[1].stamps) == 1 and not motions(bundle[3])


@pytest.mark.parametrize('fault', ['owner_change', 'preempt'])
def test_preemption_or_controller_change_during_camera_check_cannot_resume(tmp_path, monkeypatch, fault):
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch)
    current = [True]
    def changed():
        if fault == 'owner_change':
            flags['generation'] = 5
        else:
            current[0] = False
    bundle[1].on_observe = changed
    result = execute(lambda: current[0])
    assert not result['ok'] and result['reason'] == 'find_marvin_interruption_stationary_unverified'
    record, = d['collision_boundary_observations']
    assert not record['safe_approach_complete'] and not record['marvin_confirmed_forward']
    assert bundle[0]._marvin_alignment_observation is None and not motions(bundle[3])


def test_clearance_wakeup_retires_stationary_camera_authority(tmp_path, monkeypatch):
    bundle, flags, d, history, floor, execute = boundary_case(tmp_path, monkeypatch, clear_after=1.)
    result = execute()
    assert result['ok'] and result['reason'] == 'find_marvin_protected_envelope_cleared'
    record, = d['collision_boundary_observations']
    assert record['marvin_confirmed_forward'] and result['camera_floor'] > floor
    assert history[-1]['protected_envelope_clear']
    r = bundle[0]
    assert r._marvin_alignment_observation is None and not motions(bundle[3])
    refused = r.execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=result['camera_floor'])
    assert not refused['ok'] and not motions(bundle[3])
