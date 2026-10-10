"""Offline regression: retained native mission-29949cca, never robot I/O.

Historical replay consumes retained certificates and verification frontiers;
synthetic capture tests are explicitly separate from recovered live history.
"""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import threading
import time

import pytest

from marvin_navigation_certificates import (EvidenceKey, ObservationCertificate,
    GeometryCertificate, SideGeometry, VisibilityClass)
from marvin_navigation_instrumentation import (Limits, PassiveBuffer, EventType,
    DiagnosticEvent, OfflineShadowConsumer, freeze, thaw, completion_snapshot, native_completion_facts)
from marvin_navigation_issuers import (IssuerContext, Issuance, Fact, SnapshotCertificates,
    issue_completion, issue_retained_completion, issue_geometry, certificates_to_policy)
from marvin_navigation_phases import Phase, Side, Watchdogs, Context, Event
from marvin_navigation_policy import WatchdogEvidence, IntentKind
from marvin_navigation_shadow import phase_policy
from marvin_navigation_shadow_runtime import Resources, ShadowRuntime

FIXTURE=Path(__file__).parent/'test_fixtures/mission_29949cca_shadow_evidence.json'
LIMITS=Limits(64,16000,100000,32)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Offline evidence repair only')
    monkeypatch.setattr(socket,'socket',forbidden)
    monkeypatch.setattr('subprocess.Popen',forbidden)
    monkeypatch.delenv('MARVIN_NAVIGATION_SHADOW_ENABLED',raising=False)


def fixture():return json.loads(FIXTURE.read_text())


def decode_stamps(value):
    """Explicit decimal wire adapter; never float conversions or stamp changes."""
    if isinstance(value,dict):
        return {k:int(v) if k.endswith('_stamp_ns') and isinstance(v,str) and v.isdecimal()
                else decode_stamps(v) for k,v in value.items()}
    if isinstance(value,list):return [decode_stamps(v) for v in value]
    return value


def restored(f):
    associations={}
    for action in f['retained_diagnostics']['actions']:
        for name in ('pre_action_target_association','jit_target_association','next_target_association'):
            a=action.get(name) or {}
            if a.get('acquisition_sequence'):associations[a['acquisition_sequence']]=a
    geometries={};observations={};geometry_rows=[]
    for row in f['native_certificates']:
        c=copy.deepcopy(row['payload']['certificate']);kind=row['certificate_type']
        if kind=='ObservationCertificate':
            c['key']=EvidenceKey(**c['key']);c['visibility']=VisibilityClass(c['visibility']);c['bbox']=tuple(c['bbox'])
            observations[row['event_index']]=ObservationCertificate(**c)
        elif kind=='GeometryCertificate':
            c['key']=EvidenceKey(**c['key']);c['blocker_xy_m']=tuple(c['blocker_xy_m'])
            a=associations.get(c['key'].sequence) or {};minimum=(a.get('route') or {}).get('blocking_obstacle_minimum_side_separation_m')
            sides=[]
            for s in c['sides']:
                s['side']=Side(s['side'])
                if s['target_xy_m'] is not None:s['target_xy_m']=tuple(s['target_xy_m'])
                s['blocker_center_separation_m']=s['lateral_separation_m']
                s['minimum_side_separation_m']=minimum if s['lateral_separation_m'] is not None else None
                s['separation_provenance']='retained same-sequence native association route' if minimum is not None else 'MISSING_RETAINED_ASSOCIATION'
                sides.append(SideGeometry(**s))
            c['sides']=tuple(sides);g=GeometryCertificate(**c)
            geometries[row['event_index']]=g;geometry_rows.append(g)
    return observations,geometries,geometry_rows


def action_evidence(index):
    f=fixture();observations,geometries,_=restored(f)
    row=decode_stamps(f['retained_diagnostics']['actions'][index]);a=f['production_actions'][index]
    planning_event=(3,8,12)[index];post_event=(7,11,16)[index]
    planning=geometries[planning_event].key
    if index==1:
        native=next(x for x in f['native_certificates'] if x['certificate_type']=='CompletionCertificate')
        jit=EvidenceKey(**native['payload']['certificate']['jit'])
        # The generic progress row omitted turn ledger membership; the real
        # native shadow completion explicitly retained that exact fact.
        row['source_stamp_consumed']=native['payload']['native_facts']['source_stamp_consumed']
    else:
        native=row['jit_lidar_evidence']
        # Retained effective age is an actual conservative upper bound on
        # receipt age. Its provenance is explicit; no zero age is fabricated.
        jit=EvidenceKey(f['mission_id'],native['producer_session'],native['acquisition_sequence'],
            native['received_monotonic_seconds'],native['effective_age_seconds'],'RETAINED_EFFECTIVE_AGE_UPPER_BOUND')
    g=geometries[post_event];o=observations[post_event]
    ctx=IssuerContext(f['mission_id'],g.key.producer_session,max(o.key.received_monotonic_seconds,g.key.received_monotonic_seconds),
        mission_owned=True,producer_running=True)
    return row,a['primitive'],planning,jit,g,o,ctx


def replay_completions():
    results=[]
    for i in range(3):
        row,primitive,plan,jit,g,o,ctx=action_evidence(i)
        results.append(issue_retained_completion(row,primitive,plan,jit,g,o,ctx,source_mission_id=ctx.mission_id))
    return results


def event(kind,index,payload,when=1.):
    return DiagnosticEvent(1,'offline',EventType(kind),index,when,'OFFLINE_SYNTHETIC_CAPTURE',freeze(payload,LIMITS))


def capture(buffer,kind='OBSERVATION_RECORDED',payload=None,**kw):
    return buffer.capture(kind,mission_id='offline',event_time=1.,provenance='OFFLINE_SYNTHETIC_CAPTURE',payload=payload or {},**kw)


def test_exact_live_failure_chronology_reproduced():
    f=fixture();c=f['strafe_chronology']
    assert c['geometry_precedes_confirmed_zero_ms']==pytest.approx(13.260532985441387)
    from marvin_navigation_issuers import chronology, EvidenceError
    row,_,plan,jit,g,o,_=action_evidence(0)
    with pytest.raises(EvidenceError) as failure:
        chronology(used_stamp=row['authorizing_camera']['source_frame_stamp_ns'],
            observation_stamp=row['authorizing_camera']['source_frame_stamp_ns'],planning_key=plan,jit_key=jit,
            post_key=g.key,post_camera=o,stop_time=row['bridge_zero_verified_monotonic_seconds'],stop_floor=jit.sequence)
    assert failure.value.failure.field=='STOP_order'


@pytest.mark.parametrize('index',[0,1,2])
def test_retained_native_completion(index):
    result=replay_completions()[index]
    assert result.ok, result.failure
    c=result.certificate
    c.require_completed(mission_id=c.planning.mission_id,session=c.planning.producer_session)
    assert c.chronology.dispatch_monotonic_seconds is None
    assert c.chronology.automatic_stop_issued_monotonic_seconds is None
    assert c.stopped_monotonic_seconds==fixture()['production_actions'][index]['automatic_stop_completion_monotonic']


def test_valid_geometry_before_bridge_poll():
    c=replay_completions()[0].certificate
    assert c.stopped_monotonic_seconds < c.outcome_geometry.key.received_monotonic_seconds < c.chronology.bridge_zero_confirmed_monotonic_seconds
    assert c.outcome_geometry.key.received_monotonic_seconds-c.outcome_geometry.key.age_at_receipt_seconds>c.stopped_monotonic_seconds


@pytest.mark.parametrize('fault',['pre_stop','pre_action','acquired_during_motion','stale','old_camera','session','missing_zero','bad_ack','interrupted'])
def test_native_completion_rejects_unsafe_evidence(fault):
    row,primitive,plan,jit,g,o,ctx=action_evidence(0)
    if fault=='pre_stop':g=replace(g,key=replace(g.key,received_monotonic_seconds=row['stopped_monotonic_seconds']-.001))
    if fault=='pre_action':g=replace(g,key=plan)
    if fault=='acquired_during_motion':g=replace(g,key=replace(g.key,age_at_receipt_seconds=.2))
    if fault=='stale':ctx=replace(ctx,now=ctx.now+1.)
    if fault=='old_camera':o=replace(o,source_stamp_ns=row['authorizing_camera']['source_frame_stamp_ns'])
    if fault=='session':jit=replace(jit,producer_session='other')
    if fault=='missing_zero':row.pop('bridge_zero_verified_monotonic_seconds')
    if fault=='bad_ack':row['command']['bridge_acknowledgement']['automatic_stop']=False
    if fault=='interrupted':row['interrupted']=True
    r=issue_retained_completion(row,primitive,plan,jit,g,o,ctx,source_mission_id=ctx.mission_id)
    assert not r.ok


def test_live_metric_values_are_distinct():
    f=fixture();_,g,_=restored(f)
    for event_index,minimum,center in ((12,.1596218010274496,.17472024925049875),(16,.14193916196230794,.15543651943311185)):
        side=next(s for s in g[event_index].sides if s.side==Side.LEFT)
        assert side.minimum_side_separation_m==minimum
        assert side.blocker_center_separation_m==center
    assert f['production_actions'][2]['jit']['minimum_side_separation_m']==pytest.approx(.16510011806636618)


def live_post_policy(*,repair_feasible=None):
    row,primitive,plan,jit,g,o,ctx=action_evidence(2)
    g=replace(g,sides=tuple(replace(s,lateral_feasible=repair_feasible if s.side==Side.LEFT else False) for s in g.sides))
    obs=Issuance(o,None,(Fact('horizontal_error',0.,'controlled test calibration'),Fact('centering_tolerance',20.,'controlled test calibration')))
    geometry=Issuance(g,None,(Fact('LEFT.pass_permitted',True,'retained native local bypass verdict'),))
    completion=replay_completions()[2]
    data=certificates_to_policy(SnapshotCertificates(obs,geometry,completion=completion),ctx,
        watchdog=WatchdogEvidence(prior_phase=Phase.PASS_OBSTACLE,committed_side=Side.LEFT),
        delivery_resolved=True,event=Event.ACTION_COMPLETED)
    context=Context(phase=Phase.PASS_OBSTACLE,committed_side=Side.LEFT,producer_session=ctx.producer_session,
        last_event_at=ctx.now-.1,phase_entered_at=ctx.now-.1)
    return data,phase_policy(data,Watchdogs(.16),context=context)


def test_native_post_crossing_without_same_scan_repair_fact_reverifies():
    data,result=live_post_policy()
    assert data.left.separation_m==pytest.approx(.14193916196230794)
    assert result.intent.kind==IntentKind.STOP_REVERIFY
    assert result.intent.motion_authority is False


def test_pass_to_clear_side_at_live_metric_with_supported_repair_fact():
    # Repair feasibility is CONTROLLED, not invented live history. The live
    # record lacks a same-scan strafe probe. Future hooks revalidate that probe
    # on the captured native post scan on the worker (separate test below).
    _,result=live_post_policy(repair_feasible=True)
    assert result.context.phase==Phase.CLEAR_SIDE
    assert result.intent.kind==IntentKind.STRAFE_LEFT
    assert result.intent.motion_authority is False


def test_issuer_names_metrics_and_revalidates_stopped_scan_probe():
    from test_marvin_local_bypass import scene_plan, ASSOCIATION
    from marvin_local_bypass import plan_local_bypass
    lidar,selection=scene_plan()
    association=dict(ASSOCIATION,producer_session='test',acquisition_sequence=lidar['acquisition_sequence'],route=selection['route'])
    association['local_bypass_candidates']={side:plan_local_bypass(lidar,association,selection['route'],expected_session='test',side=side) for side in ('LEFT','RIGHT')}
    ctx=IssuerContext('offline','test',lidar['received_monotonic_seconds'],mission_owned=True,producer_running=True)
    association['route']['blocking_obstacle_minimum_side_separation_m']=.14193916196230794
    g=issue_geometry(lidar,association,{},ctx,source_mission_id='offline',lateral_probe_durations={'LEFT':1.})
    assert g.ok,g.failure
    side=next(s for s in g.certificate.sides if s.side==Side.LEFT)
    assert side.lateral_feasible is True
    assert side.minimum_side_separation_m==pytest.approx(.14193916196230794)
    assert side.blocker_center_separation_m==.25
    assert not side.pass_feasible(lidar['acquisition_sequence'])


def test_retained_certificate_coverage_and_calibration():
    f=fixture();o,g,rows=restored(f)
    assert len(o)==4 and len(rows)==8
    assert o[7].visibility==VisibilityClass.CALIBRATION_REQUIRED
    alignment=next(x for x in f['live_shadow_rows'] if x['event_type']=='SHADOW_DECISION' and x['event_index']==8)['payload']['alignment']
    assert alignment['horizontal_error']==58.
    assert alignment['visibility_calibration']=='CALIBRATION_REQUIRED'
    assert [a['primitive'] for a in f['production_actions'] if a['physical_dispatch_confirmed']]==['STRAFE_LEFT','TURN_RIGHT','BYPASS_FORWARD']


def test_exact_event_13_gap_in_retained_stream():
    f=fixture();indexes={r['event_index'] for r in f['live_shadow_rows']}
    assert set(range(1,18))-indexes=={13}
    assert any('EVENT_GAP' in x['reason'] for x in f['live_failures'])


def test_gap_preserves_independently_bound_native_correlation():
    c=OfflineShadowConsumer(limits=LIMITS,phase_config=Watchdogs(.16))
    c.process(event('MISSION_BEGIN',1,{}));c.plans[123]={'bound':'original'};c.pending[123]={'bound':'completion'}
    c.process(event('STOPPED_LIDAR_RECEIVED',3,{}))
    assert 123 in c.plans and 123 in c.pending
    assert c.stream_gaps==1 and c.failures==0 and c.accounting_incomplete
    assert any(r.event_type==EventType.STREAM_INCOMPLETE for r in c.records)


def test_gap_with_required_completion_missing_still_fails():
    c=OfflineShadowConsumer(limits=LIMITS,phase_config=Watchdogs(.16))
    c.process(event('MISSION_BEGIN',1,{}))
    c.process(event('STRICT_OBSERVATION_ACCEPTED',3,dict(previous_source_frame_stamp_ns=123,observation={},producer_session='test')))
    failures=[thaw(r.payload) for r in c.records if r.event_type==EventType.CERTIFICATE_BUILD_FAILED]
    assert any('Missing required ACTION_STOPPED' in f['reason'] for f in failures)
    assert c.completion_unresolved


@pytest.mark.parametrize('missing_action',[False,True])
def test_completion_pipeline_after_low_priority_gap(monkeypatch,missing_action):
    """Synthetic complete capture with real chronology; not recovered history."""
    import marvin_navigation_issuers as issuers
    f=fixture();observations,geometries,_=restored(f)
    row,primitive,plan,jit,g,o,ctx=action_evidence(0)
    state={'post':False}
    def observation(*args,**kwargs):
        return Issuance(observations[7 if state['post'] else 2],None,
            (Fact('horizontal_error',0.,'synthetic calibration'),Fact('centering_tolerance',20.,'synthetic calibration')))
    def geometry(*args,**kwargs):
        native=geometries[7 if state['post'] else 3]
        native=replace(native,sides=tuple(replace(s,lateral_feasible=True) for s in native.sides))
        return Issuance(native,None,(Fact('LEFT.pass_permitted',True,'synthetic complete capture'),))
    monkeypatch.setattr(issuers,'issue_observation',observation);monkeypatch.setattr(issuers,'issue_geometry',geometry)
    c=OfflineShadowConsumer(limits=Limits(4,16000,100000,32),phase_config=Watchdogs(.16))
    bridge=f['retained_diagnostics']['mission_stop']['bridge_after_stop']
    def emit(kind,index,payload,when):
        c.process(DiagnosticEvent(1,ctx.mission_id,EventType(kind),index,when,
            'SYNTHETIC_COMPLETE_CAPTURE_WITH_LIVE_CHRONOLOGY',freeze(payload,LIMITS)))
    base=dict(producer_session=ctx.producer_session,producer_running=True,mission_owned=True,delivery_resolved=True,bridge=bridge)
    stamp=row['authorizing_camera']['source_frame_stamp_ns']
    emit('MISSION_BEGIN',1,{},plan.received_monotonic_seconds)
    emit('STRICT_OBSERVATION_ACCEPTED',2,dict(base,observation={},geometry_pair=({},{})),plan.received_monotonic_seconds)
    emit('GEOMETRY_EVALUATED',3,dict(base,lidar={},association={},source_frame_stamp_ns=stamp,
        old_outcome={'action_type':primitive}),plan.received_monotonic_seconds)
    if not missing_action:
        ack=row['command']['bridge_acknowledgement']
        native=dict(ok=True,execution_authorized=True,motion_executed=True,full_step_completed=True,
            lateral_step={'lateral_result':ack},stop_result={'ok':True})
        emit('ACTION_STOPPED',4,dict(result=native,primitive=primitive,source_frame_stamp_ns=stamp,
            source_stamp_consumed=True,bridge=bridge,stopped_monotonic_seconds=row['bridge_zero_verified_monotonic_seconds'],
            jit_pair=(dict(producer_session=jit.producer_session,acquisition_sequence=jit.sequence,
                received_monotonic_seconds=jit.received_monotonic_seconds,age_at_receipt_seconds=jit.age_at_receipt_seconds),None),
            jit_sequence=jit.sequence,command_evidence=(dict(start_monotonic_seconds=row['command']['start_monotonic_seconds']),
                dict(completion_monotonic_seconds=row['command']['completion_monotonic_seconds'],bridge_acknowledgement=ack))),
            row['bridge_zero_verified_monotonic_seconds'])
    # Low-priority producer 5 was lost; all required action facts stay bound.
    emit('STOPPED_LIDAR_RECEIVED',6,{},ctx.now)
    state['post']=True
    emit('STRICT_OBSERVATION_ACCEPTED',7,dict(base,observation={},geometry_pair=({},{}),previous_source_frame_stamp_ns=stamp),ctx.now)
    completed=[r for r in c.records if r.event_type==EventType.ACTION_COMPLETED]
    assert bool(completed) is not missing_action
    assert c.stream_gaps>0 and c.accounting_incomplete
    if not missing_action:
        assert c.failures==0
        assert c.context.phase==Phase.PASS_OBSTACLE
        assert sum(r.event_type==EventType.STRICT_OBSERVATION_ACCEPTED for r in c.records)==2


def test_correlation_eviction_is_explicit():
    c=OfflineShadowConsumer(limits=Limits(1,100,1000,10),phase_config=Watchdogs(.16))
    c.limited_put(c.pending,123,{});c.limited_put(c.pending,456,{})
    assert c.correlation_evictions==1
    assert c.correlation_eviction_records[0]==dict(reason='correlation_eviction',source_stamp_ns=123,cache='pending_completion')


def test_native_ack_normalization_keeps_chronology():
    row,primitive,_,jit,_,_,_=action_evidence(2)
    ack=row['command']['bridge_acknowledgement']
    p=dict(primitive=primitive,result=dict(ok=True,motion_executed=True,execution_authorized=True,full_step_completed=True,
        approach_result={'forward_result':dict(ack,executed=True)},stop_result={'ok':True}),
        source_frame_stamp_ns=row['authorizing_camera']['source_frame_stamp_ns'],source_stamp_consumed=True,
        bridge=fixture()['retained_diagnostics']['mission_stop']['bridge_after_stop'],
        stopped_monotonic_seconds=row['bridge_zero_verified_monotonic_seconds'],jit_sequence=jit.sequence,
        jit_pair=({'received_monotonic_seconds':jit.received_monotonic_seconds},None),
        command_evidence=({'start_monotonic_seconds':row['command']['start_monotonic_seconds']},
            {'completion_monotonic_seconds':row['command']['completion_monotonic_seconds'],'bridge_acknowledgement':ack}))
    facts=native_completion_facts(p)
    assert facts['chronology']['automatic_stop_completed_monotonic_seconds']==row['command']['completion_monotonic_seconds']


def test_terminal_displaces_critical_geometry_when_entire_queue_full():
    b=PassiveBuffer(Resources().limits,priority=True,reserve=8)
    for _ in range(32):assert capture(b,'GEOMETRY_EVALUATED')
    assert capture(b,'TERMINAL')
    assert sum(e.event_type==EventType.TERMINAL for e in b.drain())==1
    assert b.statistics()['drop_reasons']['priority_eviction']==1


def test_terminal_retained_under_full_emergency_lane_contention():
    b=PassiveBuffer(Resources().limits,priority=True,reserve=8)
    with b._lock:
        for _ in range(8):assert capture(b,'GEOMETRY_EVALUATED')
        assert capture(b,'TERMINAL')
    assert sum(e.event_type==EventType.TERMINAL for e in b.drain())==1
    assert b.statistics()['drop_reasons']['priority_eviction']==1


def test_writer_rotation_preserves_bounded_limits(tmp_path):
    from marvin_navigation_shadow_runtime import RotatingJsonl
    resources=Resources(file_bytes=128,file_count=2,max_record_bytes=128)
    writer=RotatingJsonl(str(tmp_path),resources)
    try:
        for index in range(10):writer.write((json.dumps({'index':index,'diagnostic':'x'*60})+'\n').encode())
    finally:writer.close()
    files=list(tmp_path.glob('*.jsonl'))
    assert len(files)==2 and all(p.stat().st_size<=128 for p in files)
    assert all(json.loads(line)['index']>=8 for p in files for line in p.read_text().splitlines())


def test_slow_normal_snapshot_cannot_reorder_admitted_indexes(monkeypatch):
    import marvin_navigation_instrumentation as instrumentation
    b=PassiveBuffer(Resources().limits,priority=True,reserve=8)
    original=instrumentation.freeze;started=threading.Event();release=threading.Event()
    def slow(value,*args,**kwargs):
        if value.get('slow'):
            started.set();assert release.wait(2.)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(instrumentation,'freeze',slow)
    t=threading.Thread(target=lambda:capture(b,payload={'slow':True}));t.start()
    try:
        assert started.wait(2.)
        assert capture(b,'TERMINAL')
        first=b.drain();assert first[0].event_index==1
    finally:release.set();t.join(2.)
    assert b.drain()[0].event_index==2


def test_drop_reasons_and_nonblocking_critical_contention():
    b=PassiveBuffer(Limits(8,100,1000,10),priority=True,reserve=2)
    with b._lock:
        assert not capture(b)
        assert capture(b,'ACTION_STOPPED')
        assert capture(b,'TERMINAL')
    assert [e.event_type for e in b.drain()]==[EventType.ACTION_STOPPED,EventType.TERMINAL]
    assert not capture(b,payload={'bad':object()})
    b.close();assert not capture(b,'TERMINAL')
    s=b.statistics()
    assert s['drop_reasons']['lock_contention']==1
    assert s['drop_reasons']['invalid_snapshot']==1
    assert s['drop_reasons']['payload_rejection']==1
    assert s['drop_reasons']['shutdown_rejection']==1
    assert s['queued']==0 and s['high_water']<=8


@pytest.mark.parametrize('kind',['ACTION_STOPPED','TERMINAL','JIT_VETO_RECORDED','BLOCKED_WAIT','STRICT_OBSERVATION_ACCEPTED','GEOMETRY_EVALUATED'])
def test_critical_event_displaces_normal_priority(kind):
    b=PassiveBuffer(Limits(8,100,1000,10),priority=True,reserve=2)
    for _ in range(6):assert capture(b)
    assert capture(b,kind)
    assert any(e.event_type.value==kind for e in b.drain())
    assert b.statistics()['drop_reasons']['priority_eviction']==1


def test_bounded_full_and_reasoned_cloud_budget():
    b=PassiveBuffer(Limits(8,100,1000,10),priority=True,reserve=2,max_points=1)
    for _ in range(8):assert capture(b,'TERMINAL')
    assert not capture(b,'TERMINAL')
    assert not capture(b,payload={'points':[{},{}]})
    assert b.statistics()['drop_reasons']['queue_full']==1
    assert b.statistics()['drop_reasons']['cloud_budget']==1
    assert len(b.drain())==8


def test_terminal_retention_and_correlation_retirement():
    c=OfflineShadowConsumer(limits=LIMITS,phase_config=Watchdogs(.16))
    c.process(event('MISSION_BEGIN',1,{}));c.pending[123]={};c.plans[123]={}
    terminal={k:v for k,v in fixture()['retained_diagnostics']['terminal'].items() if k in ('state','reason','ended_monotonic_seconds')}
    c.process(event('TERMINAL',2,terminal))
    assert any(r.event_type==EventType.TERMINAL for r in c.records)
    assert not c.pending and not c.plans


def test_completion_capture_omits_large_duplicate_planner_trees():
    raw=dict(ok=True,execution_authorized=True,motion_executed=True,full_step_completed=True,
        local_detour={'duplicate':'x'*100000},approach_result={'forward_result':{'ok':False,'error':'bad'},'local_detour':{'duplicate':'x'*100000}},stop_result={'ok':True})
    compact=completion_snapshot(raw)
    assert 'local_detour' not in compact and 'local_detour' not in compact['approach_result']
    assert compact['approach_result']['forward_result']['error']=='bad'
    assert raw['local_detour']['duplicate']=='x'*100000


def test_degraded_health_does_not_disable_writer(tmp_path):
    s=ShadowRuntime(str(tmp_path));s.healthy=True
    assert s.status()['health_state']=='HEALTHY'
    assert not capture(s.buffer,payload={'bad':object()})
    h=s.status();assert h['health_state']=='DEGRADED' and not h['healthy'] and h['runtime_writer_healthy']
    assert h['enabled'] and h['drops_present'] and h['coverage_degraded'] and not h['certificate_errors_present']
    s.certificate_errors=2;s.processing_errors=2
    assert s.status()['certificate_errors_present']
    s._disable(OSError());assert s.status()['health_state']=='FAILED'


def test_cycle_bursts_and_forced_contention():
    b=PassiveBuffer(Resources().limits,priority=True,reserve=8)
    critical=('STRICT_OBSERVATION_ACCEPTED','GEOMETRY_EVALUATED','ACTION_STOPPED','JIT_VETO_RECORDED','TERMINAL')
    retained=[]
    for cycle in range(100):
        for kind in ('OBSERVATION_RECORDED','GEOMETRY_EVALUATED','SHADOW_COMPARISON','ACTION_STOPPED','STOPPED_LIDAR_RECEIVED','ROUTE_REASSESSED','STRICT_OBSERVATION_ACCEPTED','SHADOW_COMPARISON'):
            assert capture(b,kind,payload={'cycle':cycle})
        with b._lock:
            assert not capture(b,'SHADOW_COMPARISON')
            for kind in critical:assert capture(b,kind,payload={'cycle':cycle})
        retained.extend(b.drain())
    assert sum(e.event_type==EventType.TERMINAL for e in retained)==100
    assert sum(e.event_type==EventType.ACTION_STOPPED for e in retained)==200
    assert b.statistics()['drop_reasons']['lock_contention']==100
    assert b.statistics()['full_drops']==0 and b.statistics()['high_water']<=32


def test_worker_writer_terminal_jsonl_integrity(tmp_path):
    s=ShadowRuntime(str(tmp_path));s.start()
    try:
        assert capture(s.buffer,'MISSION_BEGIN')
        assert capture(s.buffer,'TERMINAL',payload={'state':'STOPPED','reason':'offline cancellation'})
        deadline=time.monotonic()+3
        while s.events_written<3 and time.monotonic()<deadline:time.sleep(.01)
        assert s.events_written>=3
    finally:s.shutdown()
    rows=[json.loads(l) for p in tmp_path.glob('*.jsonl') for l in p.read_text().splitlines()]
    assert all(r['motion_authority'] is False for r in rows)
    assert any(r['event_type']=='TERMINAL' and not r['payload'].get('capture_only') for r in rows)
    assert [r['writer_record_index'] for r in rows]==list(range(1,len(rows)+1))


@pytest.mark.parametrize('shadow',[False,True])
def test_production_equivalence_with_diagnostic_capture(tmp_path,monkeypatch,shadow):
    from test_marvin_local_bypass import mission_bundle
    from test_find_marvin_closed_loop import run,motions
    import marvin_navigation_shadow_runtime as sr
    captured=[]
    def factory():
        s=sr.ShadowRuntime(str(tmp_path/'shadow'));captured.append(s);return s
    monkeypatch.setattr(sr,'from_environment',factory)
    monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_ENABLED','true' if shadow else 'false')
    (runtime,_,_,events,_),_,_=mission_bundle(tmp_path,monkeypatch,bypass_steps=2)
    result=run(runtime)
    assert result['state']=='ARRIVED'
    assert (result['local_avoidance_actions'],result['local_bypass_actions'])==(3,2)
    assert motions(events)[:3]==[('strafe',.08,1.),('forward',.1,.5),('forward',.1,.5)]
    assert len(runtime._marvin_alignment_consumed_source_frame_stamps)==len(motions(events))
    if captured:
        kinds={e.event_type for e in captured[0].buffer.drain()}
        assert EventType.TERMINAL in kinds
        captured[0].shutdown()
