"""Offline preference policy and mission progress; all sensors/transport are fakes."""
import socket

import pytest

from marvin_detour_watchdog import MATERIAL_GEOMETRY_CHANGE_M
from marvin_local_obstacle_avoidance import rank_marvin_escape_options
from marvin_obstacle_phases import plan_phase_action
from test_marvin_avoidance_planner_progress import scan, forecast
from test_marvin_local_bypass import ASSOCIATION, OPEN_LEFT, pursuit_specs
from test_marvin_lateral_avoidance import LEFT_OPEN, strafe_runtime
from test_find_marvin_closed_loop import motions, run
from test_marvin_final_jit_action_switch import lateral_switch_mission, refused_row
from test_marvin_dispatch_guard_repair import final_boundary_mission


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Preference tests cannot access live hardware or services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def mirrored(points, side):
    return [(x, y if side == 'LEFT' else -y) for x, y in points]


def plan(points, side='LEFT', outcome=None, **kwargs):
    state = scan(mirrored(points, side));state['acquisition_sequence'] = 10
    return plan_phase_action(state, ASSOCIATION, expected_session='test',
        allow_strafe=True, committed_side=side, lateral_progress=outcome, **kwargs)


def insufficient(side='LEFT', **kwargs):
    return dict(completed_strafe=True, producer_session='test', side=side,
        acquisition_sequence=9, meaningful_progress=False, **kwargs)


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
@pytest.mark.parametrize('turn_gain', [.014, .06, .20])
def test_safe_useful_strafe_precedes_larger_turn_prediction(side, turn_gain):
    options = {'STRAFE_'+side:forecast(.014), 'TURN_'+side:forecast(turn_gain)}
    assert rank_marvin_escape_options(options, list(options)) == 'STRAFE_'+side


@pytest.mark.parametrize('fault', ['permitted', 'hard_safety_permitted', 'meaningful_progress'])
def test_ranking_never_prefers_unsafe_or_useless_strafe(fault):
    options = {'STRAFE_LEFT':forecast(.014), 'TURN_LEFT':forecast(.06)}
    if fault == 'meaningful_progress':options['STRAFE_LEFT']['route_progress'][fault] = False
    else:options['STRAFE_LEFT'][fault] = False
    assert rank_marvin_escape_options(options, list(options)) == 'TURN_LEFT'


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_fresh_lateral_swept_admission_and_useful_geometry_choose_strafe(side):
    p = plan([(.65,-.14),(0,1.2),(0,-.65)],side)
    assert p['phase'] == 'CLEAR_SIDE' and p['action_type'] == 'STRAFE_'+side
    option = p['options'][p['action_type']]
    assert option['permitted'] and option['hard_safety_permitted'] and option['improves_route']
    assert option['requested_duration'] <= 1.
    assert p['committed_side'] == side


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_commitment_outweighs_better_other_side_rank(side):
    p = plan([(.55,0),(0,.48),(0,-1.2)],side)
    assert p['action_type'] == 'STRAFE_'+side
    assert p['side_change_reason'] is None
    assert p['options']['STRAFE_'+side]['requested_duration'] < 1.


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_both_lateral_sweeps_blocked_choose_safe_same_side_turn(side):
    p = plan([(.55,0),(0,.46),(0,-.46)],side)
    assert not p['options']['STRAFE_LEFT']['permitted']
    assert not p['options']['STRAFE_RIGHT']['permitted']
    assert p['action_type'] == 'TURN_'+side and p['committed_side'] == side
    assert p['options'][p['action_type']]['permitted']
    assert p['options'][p['action_type']]['requested_duration'] == .5


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_fresh_unsafe_commitment_can_switch_to_independent_safe_strafe(side):
    p = plan([(.55,0),(0,.46),(0,-1.2)],side)
    other = 'RIGHT' if side == 'LEFT' else 'LEFT'
    assert p['action_type'] == 'STRAFE_'+other and p['committed_side'] == other
    assert p['side_change_reason'] == 'committed_lateral_unsafe'
    assert p['options'][p['action_type']]['hard_safety_permitted']


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_failed_measured_strafe_uses_turn_without_reversing(side):
    p = plan([(.55,0),(0,1.2),(0,-1.2)],side,
        insufficient(side, alternative_clearance_m=2.))
    assert p['options']['STRAFE_'+side]['permitted']
    assert p['options']['STRAFE_'+side]['improves_route']
    assert p['action_type'] == 'TURN_'+side
    assert p['reason'] == 'find_marvin_phase_lateral_progress_insufficient'
    assert p['side_change_reason'] is None


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
@pytest.mark.parametrize('gain,switch', [(.005,False),(.03,True)])
def test_ineffective_side_reversal_also_requires_material_alternative_opening(side,gain,switch):
    points = [(.55,0),(0,1.2),(0,-1.2)]
    other = 'RIGHT' if side == 'LEFT' else 'LEFT'
    clearance = plan(points,side)[other.lower()+'_clearance_m']
    p = plan(points,side,insufficient(side, alternative_clearance_m=clearance-gain))
    assert p['committed_side'] == (other if switch else side)
    assert p['action_type'] == ('STRAFE_'+other if switch else 'TURN_'+side)
    assert p['side_change_reason'] == ('committed_lateral_ineffective' if switch else None)


@pytest.mark.parametrize('bad', [
    {'completed_strafe':False}, {'producer_session':'other'},
    {'side':'RIGHT'}, {'acquisition_sequence':11}, {'acquisition_sequence':True},
    {'acquisition_sequence':None}, {'meaningful_progress':True},
])
def test_attempt_veto_incomparable_or_future_outcome_cannot_prove_ineffectiveness(bad):
    outcome = insufficient();outcome.update(bad)
    p = plan([(.65,-.14),(0,1.2),(0,-.65)],outcome=outcome)
    assert p['action_type'] == 'STRAFE_LEFT'


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_established_safe_passage_wins_even_after_insufficient_lateral_outcome(side):
    p = plan(OPEN_LEFT,side,insufficient(side))
    assert p['action_type'] == 'BYPASS_FORWARD' and p['phase'] == 'PASS_OBSTACLE'
    assert p['committed_side'] == side and p['local_bypass']['bypass_forward_permitted']


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_no_safe_turn_or_lateral_remains_blocked(side):
    p = plan([(.55,0),(0,.44),(0,-1.2)],side)
    assert p['phase'] == 'BLOCKED_WAIT' and p['action_type'] is None


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_measured_strafe_gain_keeps_side_then_enters_pass(tmp_path,monkeypatch,side):
    scenes = [mirrored(p,side) for p in [
        [(.65,-.11),(0,1.2),(0,-.65)], [(.65,-.14),(0,1.2),(0,-.65)], OPEN_LEFT]]+[None]
    b,_,_ = strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0,1.1)]*4),scenes)
    result = run(b[0])
    assert result['state'] == 'ARRIVED',result['reason']
    rows = result['local_avoidance_history']
    assert [row['selection']['action_type'] for row in rows] == ['STRAFE_'+side,'STRAFE_'+side,'BYPASS_FORWARD']
    evidence = rows[1]['selection']['lateral_progress']
    assert evidence['completed_strafe'] and evidence['meaningful_progress']
    assert evidence['center_separation_gain_m'] == pytest.approx(.03)
    assert evidence['acquisition_sequence'] > rows[0]['selection']['acquisition_sequence']
    assert rows[2]['selection']['phase'] == 'PASS_OBSTACLE'
    assert all(row['selection']['committed_side'] == side for row in rows)


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_no_measured_strafe_gain_turns_are_bounded_and_receive_no_progress(tmp_path,monkeypatch,side):
    b,_,_ = strafe_runtime(tmp_path,monkeypatch,[(0,.6)]*8,[mirrored(LEFT_OPEN,side)])
    result = run(b[0])
    assert result['reason'] == 'find_marvin_clear_side_stagnation_exhausted'
    assert motions(b[3]) == [('strafe',.08 if side == 'LEFT' else -.08,1.)]+[('turn',side,.25,.5)]*3
    assert result['local_avoidance_actions'] == 4
    watchdog = result['detour_context']['watchdog']
    assert watchdog['phase_actions'] == 4 and watchdog['phase_stagnation_count'] == 4
    assert watchdog['total_detour_actions'] == 4
    assert watchdog['last_progress'] == 'avoidance_turn_no_traversal_credit'
    assert not watchdog['pending_reassessment']
    rows = result['local_avoidance_history']
    assert all(row['selection']['lateral_progress']['meaningful_progress'] is False for row in rows[1:])
    assert all(row['selection']['committed_side'] == side for row in rows)
    assert result['detour_context']['phase_action_count'] == 1 # Only actual traversal.
    assert 'stop' in b[3] and b[2].status()['motion']['linear_x'] == 0


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_necessary_target_alignment_precedes_pursuit_without_reversal(tmp_path,monkeypatch,side):
    error = -100 if side == 'LEFT' else 100
    b,_,_ = strafe_runtime(tmp_path,monkeypatch,
        pursuit_specs([(error,1.1),(0,1.1),(0,1.1)]),
        [mirrored(OPEN_LEFT,side),mirrored(OPEN_LEFT,side),mirrored(OPEN_LEFT,side),None])
    result = run(b[0]);history = result['history']
    assert history[0]['state'] == 'ALIGNING'
    assert motions(b[3])[0] == ('turn',side,.25,.5)
    assert result['local_avoidance_history'][0]['selection']['action_type'] == 'STRAFE_'+side
    assert result['local_avoidance_history'][1]['selection']['action_type'] == 'BYPASS_FORWARD'
    assert history[1]['source_frame_stamp_ns'] > history[0]['source_frame_stamp_ns']


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_final_jit_strafe_to_pass_remains_zero_transport_and_new_stamp_only(tmp_path,monkeypatch,side):
    b,f,_ = lateral_switch_mission(tmp_path,monkeypatch,direction=side)
    result = run(b[0]);refused = refused_row(result);v = refused['result']
    assert result['state'] == 'ARRIVED'
    cert = v['dispatch_guard_rejection']
    assert cert['selection_rejection_class'] == 'LOCAL_SELECTION_CHANGED_AT_FINAL_JIT'
    assert cert['requested_action_type'] == 'STRAFE_'+side
    assert cert['final_selected_action_type'] == 'BYPASS_FORWARD'
    assert not cert['transport_attempted'] and v['actions_executed'] == 0
    assert v['source_stamp_consumed'] and not refused['motion_executed']
    stamp = refused['source_frame_stamp_ns']
    next_action = next(row for row in result['history'] if row['motion_executed'] and row['source_frame_stamp_ns'] > stamp)
    assert next_action['source_frame_stamp_ns'] != stamp
    assert next_action['observation']['opencv_tracker']['quality'] >= .8
    assert sum(row['motion_executed'] for row in result['history'] if row['state']=='AVOIDING') == result['local_avoidance_actions']
    assert f['semantic_calls'] >= 1
    before = list(b[3])
    replay = b[0]._execute_single_marvin_approach(linear_speed=.1,duration=.5,
        source_frame_stamp_ns=stamp,dispatch_guard=lambda:True)
    assert not replay['motion_executed'] and replay['actions_executed'] == 0
    assert stamp in b[0]._marvin_alignment_consumed_source_frame_stamps and b[3] == before


def test_final_jit_reverse_switch_keeps_repair_phase_and_new_authority(tmp_path,monkeypatch):
    b,f,_ = final_boundary_mission(tmp_path,monkeypatch,'repair')
    result = run(b[0]);refused = refused_row(result)
    assert not refused['result']['dispatch_guard_rejection']['transport_attempted']
    assert refused['result']['source_stamp_consumed']
    assert refused['result']['actions_executed'] == 0
    assert result['state'] == 'ARRIVED'
    assert any(row.get('phase_after_veto') == 'CLEAR_SIDE' for row in result['local_avoidance_history'])
    assert any(row['motion_executed'] and row['source_frame_stamp_ns'] > refused['source_frame_stamp_ns']
        and (row['result'].get('local_detour') or {}).get('action_type') == 'STRAFE_LEFT'
        for row in result['history'])


def test_existing_material_gain_threshold_remains_unchanged():
    assert MATERIAL_GEOMETRY_CHANGE_M == .02


@pytest.mark.parametrize('side', ['LEFT','RIGHT'])
@pytest.mark.parametrize('gain,second_kind',[(.019,'TURN'),(.021,'STRAFE')])
def test_measured_clearance_criterion_uses_existing_two_centimeter_threshold(tmp_path,monkeypatch,side,gain,second_kind):
    first = [(.65,-.11),(0,1.2),(0,-.65)]
    second = [(.65,-.11-gain),(0,1.2),(0,-.65)]
    b,_,_ = strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0,1.1)]*3),
        [mirrored(first,side),mirrored(second,side),None])
    result = run(b[0]);rows = result['local_avoidance_history']
    assert result['state'] == 'ARRIVED'
    assert rows[1]['selection']['action_type'] == second_kind+'_'+side
    outcome = rows[1]['selection']['lateral_progress']
    assert outcome['center_separation_gain_m'] == pytest.approx(gain)
    assert outcome['meaningful_progress'] is (gain >= MATERIAL_GEOMETRY_CHANGE_M)
    assert rows[1]['selection']['committed_side'] == side


@pytest.mark.parametrize('side',['LEFT','RIGHT'])
def test_interrupted_strafe_does_not_establish_completed_lateral_progress(tmp_path,monkeypatch,side):
    points = mirrored([(.65,-.14),(0,1.2),(0,-.65)],side)
    b,_,_ = strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0,1.1)]*4),
        [points,points,None],interruptions=[True,False])
    result = run(b[0]);rows = result['local_avoidance_history']
    assert result['state'] == 'ARRIVED',result['reason']
    assert result['history'][0]['result']['interrupted']
    assert rows[1]['selection']['action_type'] == 'STRAFE_'+side
    assert rows[1]['selection']['lateral_progress'] is None
    assert result['local_avoidance_actions'] == 2


@pytest.mark.parametrize('side',['LEFT','RIGHT'])
@pytest.mark.parametrize('target_range',[.6,.8])
def test_neutral_direct_route_can_still_have_useful_predicted_blocker_clearance(side,target_range):
    # A short safely bounded sweep improves blocker-center separation even
    # though the direct target-relative corridor metric is not material.
    points = [(.485,-.10),(0,.48),(0,-1.2)]
    state = scan(mirrored(points,side))
    p = plan_phase_action(state,{'verified_marvin_distance_m':target_range,'target_bearing_degrees':0},
        expected_session='test',allow_strafe=True,committed_side=side,entering_detour=True)
    assert p['action_type'] == 'STRAFE_'+side
    option = p['options'][p['action_type']]
    assert option['permitted'] and not option['improves_route']
    before_y = p['route']['blocking_obstacle_y_m']
    after_y = option['predicted_route']['blocking_obstacle_y_m']
    gain = (-1 if side == 'LEFT' else 1)*(after_y-before_y)
    assert gain >= MATERIAL_GEOMETRY_CHANGE_M
    assert p['lateral_progress'] is None # Forecast is not measured credit.
