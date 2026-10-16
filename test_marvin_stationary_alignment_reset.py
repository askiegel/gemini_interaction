"""Retained episode-reset race; real Runtime/mission with offline sensing only."""
import ast
import copy
import json
import os
from pathlib import Path
import time
from types import MethodType

import pytest

import runtime as runtime_module
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_marvin_lidar_currentness_recovery import inject_planning_outage


RETAINED = json.loads((Path(__file__).parent / 'test_fixtures' /
    'marvin_0c6f3f1b_alignment_reset.json').read_text())


def episode(tmp_path, monkeypatch, *, full_prefix=False, fault=None, baseline=False):
    # Replay retained camera errors and movement ordering. Transport, image
    # matching and the clock are doubles; this is not a physical yaw simulation.
    specs = []
    if full_prefix:
        for row in RETAINED['completed_prefix']:
            specs.append((row['error_px'], .8))
            if row['action'] in RETAINED['tracker_loss_after_actions']:
                specs.append('lost')
        prior_commands = 14
    else:
        specs.append((88.5, .8))  # The retained tenth right turn.
        prior_commands = 1
    specs += [(88.5, .8), (76.5, .8), (0., .5)]
    bundle = make_runtime(tmp_path, monkeypatch, specs)
    rt, behavior, robot, events, clock = bundle
    state = inject_planning_outage(bundle, monkeypatch, after_actions=prior_commands)
    state['prior_commands'] = prior_commands
    state['reacquired'] = []
    state['before_reset'] = []
    state['after_reset'] = []
    planning = rt._active_localization_lidar_is_current
    def retained_age(**kwargs):
        session, scan = planning(**kwargs)
        if state['waiting']:
            scan['effective_age_seconds'] = RETAINED['stationary_scan_age_seconds']
        return session, scan
    rt._active_localization_lidar_is_current = retained_age
    reset = rt._reset_marvin_alignment_consensus
    def full_reset():
        state['before_reset'].append(copy.deepcopy(rt._marvin_alignment_geometry_history))
        reset()
        state['after_reset'].append(copy.deepcopy(rt._marvin_alignment_geometry_history))
    rt._reset_marvin_alignment_consensus = full_reset

    observe = behavior.observe_find_marvin_v2
    def reacquire(**kwargs):
        # Match the real stopped semantic boundary: never continue a pending
        # tracker and call it new Gemini evidence.
        behavior._clear_marvin_v2_tracker_episode()
        value = observe()
        if state['entered']:
            preview = value['preview_result']; tracker = preview.get('opencv_tracker') or {}
            floor = kwargs['minimum_source_frame_stamp_ns']
            if fault in {'duplicate', 'older', 'semantic_old'}:
                preview['identity_source_frame_stamp_ns'] = floor - (fault == 'older')
                if fault != 'semantic_old':
                    preview['source_frame_stamp_ns'] = tracker['source_frame_stamp_ns'] = floor
            if fault == 'stale_receipt':
                preview['received_monotonic_seconds'] = tracker['received_monotonic_seconds'] = time.monotonic()-1.01
            if fault == 'identity_false': preview['identity_confirmed'] = False
            if fault == 'session_identity': preview['identity_source'] = 'marvin_session_continuity'
            if fault == 'identity_bbox_mismatch':
                preview['bbox'] = dict(preview['bbox'], x1=preview['bbox']['x1']+1)
            if fault == 'wrong_episode':
                preview.update(identity_source='marvin_locked_tracker_continuity',
                    post_action_tracker_continuity=True, post_action_source_frame_stamp_ns=floor,
                    marvin_tracking_episode={'state': 'TRACKING'})
            if fault == 'quality': tracker['quality'] = .799999
            if fault == 'unmatched': tracker['matched'] = False
            if fault == 'tracker_frame_mismatch': tracker['source_frame_stamp_ns'] -= 1
            if fault == 'operator_stop': rt.submit_intent({'intent': 'STOP'})
            if fault == 'collision': behavior.unsafe_turn = True
            state['reacquired'].append(copy.deepcopy(preview))
        return value
    behavior.reacquire_find_marvin_v2 = reacquire
    if fault == 'failed_stop':
        stop = robot.stop
        robot.stop = lambda: {'ok': False} if state['waiting'] else stop()
    if fault == 'active_controller':
        status = robot.status
        def active():
            value = status()
            if state['waiting']: value['motion']['angular_z'] = .1
            return value
        robot.status = active

    if baseline:
        # Only the unmodified mission method is loaded; no module startup,
        # network, ROS, service or real robot object is instantiated.
        path = os.environ.get('MAYDAY_ALIGNMENT_BASELINE_FILE')
        if not path:
            pytest.skip('retained baseline required for counterfactual')
        tree = ast.parse(Path(path).read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CognitiveRuntime')
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
            and x.name == '_execute_normal_marvin_find_mission_locked')
        namespace = dict(vars(runtime_module))
        exec(compile(ast.Module(body=[method], type_ignores=[]), path, 'exec'), namespace)
        rt._execute_normal_marvin_find_mission_locked = MethodType(namespace[method.name], rt)
    return bundle, state


@pytest.mark.parametrize('baseline', [True, False])
@pytest.mark.parametrize('full_prefix', [False, True])
def test_retained_recovery_reproduces_old_refusal_and_new_admission(
        tmp_path, monkeypatch, baseline, full_prefix, record_property):
    bundle, state = episode(tmp_path, monkeypatch, baseline=baseline, full_prefix=full_prefix)
    rt, _, _, events, _ = bundle
    result = run(rt)
    recovery, = result['stationary_lidar_recovery_history']
    assert recovery['stop_ok'] and recovery['bridge_zero']
    assert recovery['new_sequence'] > recovery['selected_sequence_floor']
    assert recovery['retired_source_stamp_ns'] not in rt._marvin_alignment_consumed_source_frame_stamps
    if baseline:
        assert result['reason'] == RETAINED['old_terminal_reason']
        assert result['final_observation']['geometry_continuity']['reason'] == RETAINED['old_geometry_reason']
        assert result['actions_executed'] == len(motions(events)) == state['prior_commands']
        assert not result['history'][-1]['motion_executed']
    else:
        assert result['state'] == 'ARRIVED', result['reason']  # Synthetic arrival only.
        replacement = result['history'][state['prior_commands']]
        obs = replacement['observation']
        assert obs['identity_source'] == 'gemini_marvin_candidate_selection'
        assert obs['geometry_continuity']['accepted']
        assert obs['geometry_continuity']['reason'] == 'geometry_baseline_established'
        assert obs['source_frame_stamp_ns'] > obs['identity_source_frame_stamp_ns'] > recovery['retired_source_stamp_ns']
        assert replacement['result']['full_step_completed'] and replacement['result']['source_stamp_consumed']
        assert result['actions_executed'] == len(motions(events)) == state['prior_commands']+1
        assert any(entries and entries[-1]['identity_source'] == 'marvin_locked_tracker_continuity'
                   for entries in state['before_reset'])
        assert all(entries == [] for entries in state['after_reset'])
    assert result['completed_forward_actions'] == (4 if full_prefix else 0)
    assert result['interrupted_forward_attempts'] == result['interrupted_strafe_attempts'] == 0
    assert not result['safe_approach_complete']
    record_property('retained_reset_replay', json.dumps(dict(baseline=baseline,
        full_prefix=full_prefix, reason=result['reason'], actions=result['actions_executed'],
        forward_completed=result['completed_forward_actions'], recovery=recovery,
        final_geometry=result['final_observation']['geometry_continuity'])))


@pytest.mark.parametrize('fault', ['duplicate', 'older', 'semantic_old', 'stale_receipt',
    'identity_false', 'session_identity', 'identity_bbox_mismatch', 'wrong_episode',
    'quality', 'unmatched', 'tracker_frame_mismatch',
    'operator_stop', 'collision', 'failed_stop', 'active_controller'])
def test_reset_never_grants_motion_without_current_identity_and_safety(
        tmp_path, monkeypatch, fault):
    bundle, state = episode(tmp_path, monkeypatch, fault=fault)
    result = run(bundle[0])
    assert result['state'] != 'ARRIVED'
    assert len(motions(bundle[3])) == 1
    assert 'stop' in bundle[3]
    if fault == 'operator_stop':
        assert result['behavior'] == 'STOP'
        assert bundle[0].get_status()['runtime_state'] == 'STOPPED'
        assert bundle[0]._marvin_alignment_observation is None
        return
    assert result['actions_executed'] == 1
    assert result['completed_forward_actions'] == result['interrupted_forward_attempts'] == 0
    assert all(not h['motion_executed'] for h in result['history'][1:])
    assert not result['safe_approach_complete']


def test_source_change_inside_unretired_episode_still_rejects(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(88.5, .8), (76.5, .8)])
    rt, behavior, _, events, _ = bundle
    first = rt.observe_find_marvin_v2()
    # Source-change protection is unchanged outside a confirmed recovery reset.
    rt._marvin_alignment_geometry_history[0]['identity_source'] = 'marvin_locked_tracker_continuity'
    second = rt.observe_find_marvin_v2()
    assert first['geometry_continuity']['accepted']
    assert second['geometry_continuity']['reason'] == 'geometry_identity_source_changed'
    assert rt._marvin_alignment_observation is None and not motions(events)
