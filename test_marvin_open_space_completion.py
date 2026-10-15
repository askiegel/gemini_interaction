"""Conditional open-space stopping, fake transport only; no measured arrival."""
import copy
import time

import pytest

from mission_manager import MissionManager
from test_find_marvin_closed_loop import make_runtime, motions
from test_marvin_continuous_lidar_recovery import certificate, prepare
from test_marvin_vision_collision_only import FIXTURE, offline


def scene(tmp_path, monkeypatch, *, confirmed=True, point=(.49, 0.), visual=(0, .8),
          mode='interruption', sensor_fault=None, recorded=None, stop_reason='active_lidar_intrusion',
          extra_point=None):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8), visual], calibrated_arrival=False)
    r, b, robot, events, clock = bundle
    flags = prepare(bundle, monkeypatch, clear_after=None)
    read = r.world_model.get_lidar_obstacles
    def lidar(**kw):
        value = read(**kw)
        for p in value['local_motion_geometry']['points']:
            if p['x_m'] == .3 and p['y_m'] == 0.:
                p.update(x_m=point[0], y_m=point[1])
        if flags['waiting'] and extra_point is not None:
            value['local_motion_geometry']['points'].append(dict(x_m=extra_point[0], y_m=extra_point[1]))
        if flags.get('after_camera'):
            if sensor_fault == 'stale': value['effective_age_seconds'] = .31
            if sensor_fault == 'frozen':
                value['bridge_scan_identity'] = dict(flags['before_camera_scan'])
            if sensor_fault == 'coverage':
                value['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
            if sensor_fault == 'producer': value['bridge_scan_identity']['producer_session'] = 'other-native'
            if sensor_fault == 'regression': value['bridge_scan_identity']['source_stamp_ns'] = 1
            if sensor_fault == 'world_session': value['producer_session'] = 'other-worker'
            if sensor_fault == 'native_malformed': value['bridge_scan_identity'] = [1]
        if flags.get('final_stop'):
            if flags.get('terminal_fault') == 'final_scan_invalid': value['effective_age_seconds'] = .31
            if flags.get('terminal_fault') == 'final_native_session':
                value['bridge_scan_identity']['producer_session'] = 'new-native'
            if flags.get('terminal_fault') == 'final_worker_session': r.lidar_worker.session = 'new-worker'
        if isinstance(value['bridge_scan_identity'], dict):
            flags['last_native'] = dict(value['bridge_scan_identity'])
        return value
    r.world_model.get_lidar_obstacles = lidar
    observe = b.observe_find_marvin_v2
    def camera():
        value = observe()
        if len(b.stamps) == 2:
            if recorded is not None:
                # Retained real geometry/quality, explicitly rebased onto the
                # fake camera clock. No archived frame is claimed live/fresh.
                preview = value['preview_result']
                tracker = copy.deepcopy(recorded)
                tracker.update(source_frame_stamp_ns=preview['source_frame_stamp_ns'],
                    received_monotonic_seconds=preview['received_monotonic_seconds'])
                preview.update(opencv_tracker=tracker, bbox=tracker['bbox'])
            flags['before_camera_scan'] = dict(flags['last_native'])
            flags['after_camera'] = True
        return value
    b.observe_find_marvin_v2 = camera
    if mode == 'initial':
        flags.update(waiting=True, began=time.monotonic())
    elif mode == 'jit':
        approach = r._execute_single_marvin_approach
        def jit(**kw):
            flags.update(waiting=True, began=time.monotonic())
            return approach(**kw)
        r._execute_single_marvin_approach = jit
    else:
        def forward(*, speed, seconds, dispatch_guard=None):
            assert dispatch_guard is None or dispatch_guard()
            assert not flags['waiting'], 'No second transport permitted'
            events.append(('forward', speed, seconds))
            clock[0] += 120_000_000
            flags.update(waiting=True, began=time.monotonic())
            return certificate(duration=seconds, reason=stop_reason)[0]
        robot.move_forward = forward
    def execute():
        intent = dict(intent='FIND_OBJECT', target='Marvin', speech='Offline controlled open space')
        if confirmed is not None:
            intent['marvin_open_space_confirmed'] = confirmed
        r.submit_intent(intent)
        return r.run_once()
    return bundle, flags, execute


@pytest.mark.parametrize('mode', ['initial', 'jit', 'interruption'])
def test_verified_open_space_stop_is_terminal_conditional_completion(tmp_path, monkeypatch, mode):
    bundle, flags, execute = scene(tmp_path, monkeypatch, mode=mode)
    result = execute()
    assert result['state'] == 'SAFE_APPROACH_COMPLETE', result['reason']
    assert result['ok'] and result['safe_approach_complete']
    assert result['mission_outcome'] == 'safe_approach_complete'
    assert result['completion_conditional_on'] == 'operator_confirmed_open_space'
    assert not result['arrived_at_marvin']
    proof = result['safe_approach_evidence']
    assert proof['camera_recheck_performed'] and proof['marvin_confirmed_forward']
    assert not proof['boundary_object_identity_verified'] and not proof['motion_authorized']
    assert proof['completion_lidar']['bridge_scan_identity']['acquisition_sequence'] > proof['bridge_scan_identity']['acquisition_sequence']
    assert proof['completion_lidar']['bridge_scan_identity']['source_stamp_ns'] > proof['bridge_scan_identity']['source_stamp_ns']
    assert result['bridge_after_stop']['guarded_controller']['gait_zero']
    assert not result['bridge_after_stop']['guarded_controller']['active']
    assert all(result['bridge_after_stop']['motion'][k] == 0 for k in ('linear_x','linear_y','angular_z'))
    r, b, robot, events, clock = bundle
    assert len(motions(events)) == (1 if mode == 'interruption' else 0)
    assert events[-1] == 'stop' and len(b.stamps) == 2
    before = list(events)
    r.run_once()
    assert events == before and r.get_status()['runtime_state'] == 'IDLE'
    assert r.mission_manager.get_active_mission() is None and not r.mission_manager.get_queue()
    assert r._marvin_alignment_observation is None


@pytest.mark.parametrize('confirmed', [None, False])
def test_automatic_completion_is_disabled_without_this_mission_confirmation(tmp_path, monkeypatch, confirmed):
    bundle, flags, execute = scene(tmp_path, monkeypatch, confirmed=confirmed)
    result = execute()
    assert result['state'] == 'BLOCKED' and result['mission_outcome'] == 'safe_incomplete'
    assert not result['safe_approach_complete'] and not result['arrived_at_marvin']
    assert len(motions(bundle[3])) == 1 and bundle[3][-1] == 'stop'


@pytest.mark.parametrize('point', [(0., .3), (0., -.3), (-.3, 0.), (.3, .3), (.1, 0.)])
def test_nonforward_or_inner_footprint_intrusion_cannot_complete(tmp_path, monkeypatch, point):
    bundle, flags, execute = scene(tmp_path, monkeypatch, point=point)
    result = execute()
    assert not result['safe_approach_complete'] and not result['arrived_at_marvin']
    assert result['state'] == 'BLOCKED' and len(motions(bundle[3])) == 1


@pytest.mark.parametrize('fault', ['lost', 'off_axis', 'duplicate', 'stale', 'foreground_occludes'])
def test_unexpected_foreground_or_invalid_visual_evidence_stays_incomplete(tmp_path, monkeypatch, fault):
    visual = 'lost' if fault in {'lost', 'foreground_occludes'} else (120, .8) if fault == 'off_axis' else (0, .8)
    bundle, flags, execute = scene(tmp_path, monkeypatch, visual=visual)
    b = bundle[1]
    def change():
        if len(b.stamps) == 1:
            b.repeat_camera = fault == 'duplicate'
            b.stale_camera = fault == 'stale'
    # Set camera faults only after the first valid motion observation.
    forward = bundle[2].move_forward
    def action(**kw):
        value = forward(**kw)
        change()
        return value
    bundle[2].move_forward = action
    result = execute()
    assert not result['safe_approach_complete'] and not result['arrived_at_marvin']
    assert result['state'] == 'BLOCKED' and result['mission_outcome'] == 'safe_incomplete'
    assert len(motions(bundle[3])) == 1 and bundle[3][-1] == 'stop'


@pytest.mark.parametrize('fault', ['frozen', 'stale', 'coverage', 'producer', 'regression', 'world_session', 'native_malformed'])
def test_post_camera_sensor_failure_cannot_complete(tmp_path, monkeypatch, fault):
    bundle, flags, execute = scene(tmp_path, monkeypatch, sensor_fault=fault)
    result = execute()
    assert result['state'] == 'BLOCKED' and not result['safe_approach_complete']
    assert result['mission_outcome'] == 'safe_incomplete' and len(motions(bundle[3])) == 1


@pytest.mark.parametrize('point', [(0., .3), (-.3, 0.)])
def test_front_boundary_with_another_side_or_rear_intrusion_is_incomplete(tmp_path, monkeypatch, point):
    bundle, flags, execute = scene(tmp_path, monkeypatch, extra_point=point)
    result = execute()
    assert result['state'] == 'BLOCKED' and not result['safe_approach_complete']
    assert len(motions(bundle[3])) == 1


@pytest.mark.parametrize('reason', ['active_lidar_stale', 'active_lidar_unavailable', 'active_lidar_coverage_missing'])
def test_sensing_interruption_is_not_relabelled_success(tmp_path, monkeypatch, reason):
    bundle, flags, execute = scene(tmp_path, monkeypatch, stop_reason=reason)
    result = execute()
    assert result['state'] == 'BLOCKED' and not result['safe_approach_complete']
    assert len(motions(bundle[3])) == 1


@pytest.mark.parametrize('fault', ['operator_stop_for_foreground', 'owner_change'])
def test_start_contract_never_overrides_stop_or_changed_ownership(tmp_path, monkeypatch, fault):
    bundle, flags, execute = scene(tmp_path, monkeypatch)
    r, b, robot, events, clock = bundle
    def intervene():
        if len(b.stamps) == 2:
            if fault == 'owner_change':
                flags['generation'] = 5
            else:
                r.submit_intent(dict(intent='STOP', speech='Unexpected foreground object'))
    b.on_observe = intervene
    result = execute()
    assert not result.get('safe_approach_complete') and not result.get('arrived_at_marvin')
    assert result['state'] == ('STOPPED' if fault == 'operator_stop_for_foreground' else 'BLOCKED')
    assert len(motions(events)) == 1 and events[-1] == 'stop'


@pytest.mark.parametrize('fault', ['stop_fails', 'gait_nonzero', 'camera_expires', 'scan_expires',
                                  'final_scan_invalid', 'final_native_session', 'final_worker_session'])
def test_terminal_stop_and_fresh_evidence_are_verified_again(tmp_path, monkeypatch, fault):
    bundle, flags, execute = scene(tmp_path, monkeypatch)
    r, b, robot, events, clock = bundle
    flags['terminal_fault'] = fault
    stop = robot.stop
    status = robot.status
    def final_stop():
        value = stop()
        if flags.get('after_camera'):
            flags['final_stop'] = True
            if fault == 'stop_fails': value['ok'] = False
            if fault == 'camera_expires': clock[0] += 1_100_000_000
            if fault == 'scan_expires': clock[0] += 310_000_000
        return value
    def final_status():
        value = status()
        if flags.get('final_stop') and fault == 'gait_nonzero':
            value['guarded_controller']['gait_zero'] = False
        return value
    robot.stop, robot.status = final_stop, final_status
    result = execute()
    assert result['state'] == 'BLOCKED' and not result['safe_approach_complete']
    assert not result['arrived_at_marvin'] and len(motions(events)) == 1


@pytest.mark.parametrize('recorded', [FIXTURE['refusal']['tracker'], FIXTURE['october6_arrival']['tracker']])
def test_retained_strict_tracker_geometry_can_confirm_conditional_stop(tmp_path, monkeypatch, recorded):
    bundle, flags, execute = scene(tmp_path, monkeypatch, recorded=recorded)
    result = execute()
    assert result['state'] == 'SAFE_APPROACH_COMPLETE', result['reason']
    assert result['completion_conditional_on'] == 'operator_confirmed_open_space'
    assert not result['arrived_at_marvin'] and len(motions(bundle[3])) == 1


def test_hidden_contract_violation_is_an_explicit_observability_limit(tmp_path, monkeypatch):
    # A foreground object that leaves identical Marvin/scan evidence cannot be
    # detected by declaring open space. The operator contract is trusted input;
    # the result must remain conditional and never claim object identity/range.
    bundle, flags, execute = scene(tmp_path, monkeypatch)
    result = execute()
    assert result['safe_approach_complete']
    assert result['completion_conditional_on'] == 'operator_confirmed_open_space'
    assert not result['safe_approach_evidence']['boundary_object_identity_verified']
    assert not result['arrived_at_marvin'] and len(motions(bundle[3])) == 1


@pytest.mark.parametrize('value', [None, 1, 0, 'true', {}, []])
def test_contract_field_requires_literal_boolean(value):
    manager = MissionManager()
    mission = manager.handle_intent(dict(intent='FIND_OBJECT', target='marvin',
        marvin_open_space_confirmed=value))
    assert mission.status == 'REJECTED' and manager.get_active_mission() is None


@pytest.mark.parametrize('fields', [dict(intent='MOVE_FORWARD'), dict(target='chair'),
    dict(marvin_one_step_test=True), dict(marvin_centering_test=True),
    dict(marvin_guarded_approach_test=True)])
def test_contract_cannot_apply_to_other_motion_or_test_modes(fields):
    manager = MissionManager()
    intent = dict(intent='FIND_OBJECT', target='marvin', marvin_open_space_confirmed=True)
    intent.update(fields)
    assert manager.handle_intent(intent).status == 'REJECTED'
    assert manager.get_active_mission() is None


def test_confirmation_is_serialized_per_mission_and_never_inherited():
    manager = MissionManager()
    first = manager.handle_intent(dict(intent='FIND_OBJECT', target='marvin', marvin_open_space_confirmed=True))
    manager.complete_active_mission()
    second = manager.handle_intent(dict(intent='FIND_OBJECT', target='marvin'))
    assert first.to_dict()['marvin_open_space_confirmed'] is True
    assert second.mission_id != first.mission_id
    assert second.to_dict()['marvin_open_space_confirmed'] is False


def test_duplicate_intent_cannot_retroactively_confirm_an_active_mission():
    manager = MissionManager()
    first = manager.handle_intent(dict(intent='FIND_OBJECT', target='marvin'))
    duplicate = manager.handle_intent(dict(intent='FIND_OBJECT', target='marvin', marvin_open_space_confirmed=True))
    assert duplicate is first and not duplicate.marvin_open_space_confirmed
