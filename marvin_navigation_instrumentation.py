"""Passive capture and a separate diagnostic-only consumer.

Capture NEVER calls perception, a selector, safety admission, or an executor.
Processing/serialization is not called by runtime hooks. Navigation intent is
never motion authority. The isolated shadow-runtime module owns any worker;
this module remains usable by manual offline replay. Disabled is None (no queue).
All capacities/budgets must be supplied: test choices are provisional.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import Enum
from itertools import count
import json
import math
import threading


class EventType(str, Enum):
    MISSION_BEGIN = 'MISSION_BEGIN'
    STRICT_OBSERVATION_ACCEPTED = 'STRICT_OBSERVATION_ACCEPTED'
    OBSERVATION_RECORDED = 'OBSERVATION_RECORDED'
    GEOMETRY_EVALUATED = 'GEOMETRY_EVALUATED'
    ACTION_STOPPED = 'ACTION_STOPPED'
    STOPPED_LIDAR_RECEIVED = 'STOPPED_LIDAR_RECEIVED'
    JIT_VETO_RECORDED = 'JIT_VETO_RECORDED'
    JIT_VETO_CERTIFIED = 'JIT_VETO_CERTIFIED'
    ROUTE_REASSESSED = 'ROUTE_REASSESSED'
    BLOCKED_WAIT = 'BLOCKED_WAIT'
    TERMINAL = 'TERMINAL'
    ACTION_COMPLETED = 'ACTION_COMPLETED'
    CERTIFICATE_BUILD_FAILED = 'CERTIFICATE_BUILD_FAILED'
    SHADOW_DECISION = 'SHADOW_DECISION'
    SHADOW_COMPARISON = 'SHADOW_COMPARISON'


@dataclass(frozen=True, slots=True)
class Limits:
    capacity: int
    max_nodes: int
    max_text_bytes: int
    max_depth: int
    max_integer_bits: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v<=0 for v in (self.capacity,self.max_nodes,self.max_text_bytes,self.max_depth,self.max_integer_bits)):
            raise ValueError('Explicit positive provisional resource limits required')


@dataclass(frozen=True, slots=True)
class FrozenMap:
    entries: tuple

    def __post_init__(self):
        if type(self.entries) is not tuple:raise ValueError('frozen entries')
        for pair in self.entries:
            if type(pair) is not tuple or len(pair)!=2 or type(pair[0]) is not str:raise ValueError('string key pairs')
            validate_frozen(pair[1])


def validate_frozen(value):
    if type(value) in (str,int,bool,type(None)):return
    if type(value) is float and math.isfinite(value):return
    if type(value) is FrozenMap:return
    if type(value) is tuple:
        for x in value:validate_frozen(x)
        return
    raise ValueError('Evidence data only; no clients, callbacks, mutable objects or grants')


FORBIDDEN = frozenset({'motion_authority','dispatch_callback','bridge_client','transport_handle',
    'reusable_jit_grant','command_token','callable_executor'})


def freeze(value, limits, *, omit_keys=frozenset(), max_points=None, points_xy_only=False):
    """Bounded structural snapshot. Does not call arbitrary deepcopy hooks."""
    nodes=0;text=0
    def visit(x,depth=0,point_projection=False):
        nonlocal nodes,text
        nodes+=1
        if nodes>limits.max_nodes or depth>limits.max_depth:raise ValueError('snapshot node/depth budget')
        if isinstance(x,Enum):return visit(x.value,depth+1)
        if type(x) is str:
            if len(x)>limits.max_text_bytes:raise ValueError('snapshot text budget')
            text+=len(x.encode('utf8'))
            if text>limits.max_text_bytes:raise ValueError('snapshot text budget')
            return x
        if type(x) is int:
            if x.bit_length()>limits.max_integer_bits:raise ValueError('snapshot integer budget')
            return x
        if type(x) in (bool,type(None)):return x
        if type(x) is float and math.isfinite(x):return x
        if type(x) in (list,tuple):return tuple(visit(v,depth+1,point_projection) for v in x)
        if type(x) is dict:
            pairs=[]
            for k,v in x.items():
                if type(k) is not str:raise ValueError('plain string evidence key required')
                if point_projection and k not in ('x_m','y_m'):continue
                if k in omit_keys:continue
                # Never truncate a safety cloud: an oversized native snapshot
                # is rejected, not turned into an apparently clear corridor.
                if k=='points' and max_points is not None and type(v) in (list,tuple) and len(v)>max_points:
                    raise ValueError('native cloud exceeds diagnostic budget')
                if type(k) is not str or (k in FORBIDDEN and not (k=='motion_authority' and v is False)):
                    raise ValueError('forbidden/non-string evidence key')
                visit(k,depth+1)
                if k.endswith('_stamp_ns') and v is not None and type(v) is not int:raise ValueError('exact integer stamp required')
                pairs.append((k,visit(v,depth+1,points_xy_only and k=='points')))
            return FrozenMap(tuple(pairs))
        raise ValueError('only already-produced plain evidence objects')
    return visit(value)


def thaw(value):
    if type(value) is FrozenMap:return {k:thaw(v) for k,v in value.entries}
    if type(value) is tuple:return [thaw(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    schema_version: int
    mission_id: str
    event_type: EventType
    event_index: int
    event_time: float
    provenance: str
    payload: FrozenMap

    def __post_init__(self):
        if self.schema_version!=1 or type(self.schema_version) is not int:raise ValueError('version')
        if type(self.mission_id) is not str or not self.mission_id:raise ValueError('mission binding')
        if type(self.event_type) is not EventType or type(self.event_index) is not int or self.event_index<1:raise ValueError('typed event/ordering')
        if type(self.event_time) not in (int,float) or not math.isfinite(self.event_time) or self.event_time<0:raise ValueError('monotonic time')
        if type(self.provenance) is not str or not self.provenance or type(self.payload) is not FrozenMap:raise ValueError('immutable provenance/payload')


class PassiveBuffer:
    """Non-waiting try-lock; drop NEWEST on full, contention, or bad payload.

    CPython's native itertools.count provides atomic diagnostic counters under
    the GIL; readout uses its built-in reduce state. Other interpreters need a
    separate concurrency review before use. Payload copying is resource bounded
    but has CPU cost; this prototype makes no real-time/robot latency claim.
    """
    def __init__(self,limits,*,priority=False,reserve=0,omit_keys=frozenset(),max_points=None,points_xy_only=False):
        if type(limits) is not Limits:raise ValueError('explicit limits')
        self.limits=limits;self._lock=threading.Lock();self._events=deque()
        self._indices=count(1);self._full=count();self._contention=count();self._failed=count()
        self._accepting=True;self.priority=priority;self.reserve=reserve
        self.omit_keys=omit_keys;self.max_points=max_points;self._evicted=count()
        self.points_xy_only=points_xy_only
        self._high_water=0
        if type(reserve) is not int or not 0<=reserve<limits.capacity:raise ValueError('critical reserve')

    def capture(self,kind,*,mission_id,event_time,provenance,payload):
        if not self._lock.acquire(blocking=False):next(self._indices);next(self._contention);return False
        try:
            index=next(self._indices)
            if not self._accepting:next(self._full);return False
            if (type(mission_id) is not str or len(mission_id)>128
                    or type(provenance) is not str or len(provenance)>256
                    or type(kind) not in (str,EventType)
                    or type(kind) is str and len(kind)>64):
                next(self._failed);return False
            level=event_priority(kind) if self.priority else 1
            ceiling=self.limits.capacity if level==2 else self.limits.capacity-self.reserve
            victim=None
            if len(self._events)>=ceiling:
                if self.priority and level>0:
                    victim=next((e for rank in range(level) for e in self._events
                        if event_priority(e.event_type)==rank),None)
                if victim is None:next(self._full);return False
            try:
                event=DiagnosticEvent(1,mission_id,EventType(kind),index,event_time,provenance,
                    freeze(payload,self.limits,omit_keys=self.omit_keys,max_points=self.max_points,points_xy_only=self.points_xy_only))
                if victim is not None:self._events.remove(victim);next(self._evicted)
                self._events.append(event);self._high_water=max(self._high_water,len(self._events));return True
            except Exception:next(self._failed);return False
        finally:self._lock.release()

    def drain(self,max_items=None):
        if not self._lock.acquire(blocking=False):return ()
        try:
            n=len(self._events) if max_items is None else min(max_items,len(self._events))
            return tuple(self._events.popleft() for _ in range(n))
        finally:self._lock.release()

    def statistics(self):
        # Atomic snapshots of counts; event length is diagnostic only.
        peek=lambda c:c.__reduce__()[1][0]
        return dict(capacity=self.limits.capacity,queued=len(self._events),
            full_drops=peek(self._full),contention_drops=peek(self._contention),construction_failures=peek(self._failed),
            evicted_events=peek(self._evicted),accepting=self._accepting,high_water=self._high_water)

    def close(self):
        # No lock or waiting: capture rechecks this under its own try-lock.
        # A capture already in flight may finish; the bounded drain accepts it.
        self._accepting=False


def event_priority(kind):
    """Diagnostic retention only; never consulted by navigation or safety."""
    if str(getattr(kind,'value',kind)) in {'MISSION_BEGIN','JIT_VETO_RECORDED','JIT_VETO_CERTIFIED',
        'ACTION_STOPPED','ACTION_COMPLETED','BLOCKED_WAIT','TERMINAL','SHADOW_DECISION'}:return 2
    if str(getattr(kind,'value',kind))=='SHADOW_COMPARISON':return 0
    return 1


def make_instrumentation(*,enabled=False,limits=None):
    if type(enabled) is not bool:raise ValueError('explicit feature flag')
    return PassiveBuffer(limits) if enabled else None


def passive_capture(buffer,kind,**kwargs):
    """Trusted queue only; malformed sink objects are never called."""
    try:
        if type(buffer) is PassiveBuffer:buffer.capture(kind,**kwargs)
    except Exception:
        # Diagnostic failure cannot enter the controller return path.
        if type(buffer) is PassiveBuffer:next(buffer._failed)


def serialize_event(event):
    if type(event) is not DiagnosticEvent:raise ValueError('typed event')
    return json.dumps(dict(schema_version=event.schema_version,mission_id=event.mission_id,
        event_type=event.event_type.value,event_index=event.event_index,event_time=event.event_time,
        provenance=event.provenance,payload=thaw(event.payload)),allow_nan=False,separators=(',',':'))


def native_completion_facts(payload):
    """Derive ONLY from native successful acknowledgement/STOP contracts.

    A confirmed bounded acknowledgement proves a transport attempt and receipt,
    not displacement. Missing ack or ambiguous/incomplete return stays unknown.
    Explicit nested uncertainty overrides summary success. Raw native output and
    exact ledger membership are retained alongside these diagnostic facts.
    """
    from marvin_navigation_issuers import bridge_stationary, reject_delivery_contradictions
    from behavior_manager import BehaviorManager
    r=payload['result'];ack=(r.get('lateral_step') or {}).get('lateral_result') or (r.get('approach_result') or {}).get('forward_result') or (r.get('turn_result') or {}).get('transport_result') or {}
    turn=r.get('turn_result') or {}
    if not ack and turn.get('mode')=='bounded':ack=turn
    attempted=confirmed=certain=complete=None
    reject_delivery_contradictions(r)
    bounded=(ack.get('ok') is True and ack.get('mode')=='bounded' and ack.get('automatic_stop') is True and ack.get('returned_immediately') is False)
    physical=(r.get('motion_executed') is True and r.get('ok') is True)
    if payload['primitive'] in ('FORWARD','BYPASS_FORWARD') and bounded:
        bounded=BehaviorManager._is_canonical_marvin_bounded_forward_result(ack,speed=ack.get('linear_x'),duration=ack.get('duration'))
    if bounded and physical and (r.get('stop_result') or {}).get('ok') is True and bridge_stationary(payload.get('bridge')):
        attempted=confirmed=True;certain=True
        complete=(r.get('full_step_completed') is True or payload['primitive'] in ('TURN_LEFT','TURN_RIGHT') and turn.get('confirmed_forwarded') is True)
    return dict(action_type=payload['primitive'],source_frame_stamp_ns=payload['source_frame_stamp_ns'],
        observation_source_frame_stamp_ns=payload['source_frame_stamp_ns'],execution_authorized=r.get('execution_authorized'),
        transport_attempted=attempted,transport_confirmed=confirmed,delivery_uncertain=False if certain else r.get('delivery_uncertain'),
        motion_executed=r.get('motion_executed'),full_step_completed=complete,automatic_stop=ack.get('automatic_stop'),
        source_stamp_consumed=payload.get('source_stamp_consumed'),stop_result=r.get('stop_result'),
        bridge_after_stop=payload.get('bridge'),stopped_monotonic_seconds=payload.get('stopped_monotonic_seconds'),
        stop_lidar_sequence=payload.get('jit_sequence'),
        meaningful_progress=payload.get('meaningful_progress'),meaningful_progress_reason=payload.get('meaningful_progress_reason'))


def alignment_diagnostics(observation):
    t=observation.get('opencv_tracker') or {};b=t.get('bbox');w=t.get('image_width')
    margins=None
    if type(b) is dict and type(w) is int and all(type(b.get(k)) in (int,float) for k in ('x1','x2')):
        margins={'left_px':b['x1'],'right_px':w-b['x2']}
    return dict(image_width=w,bbox=b,center_x=t.get('center_x'),horizontal_error=t.get('horizontal_error'),
        tracker_quality=t.get('quality'),tracker_threshold=t.get('threshold'),edge_margins=margins,
        visibility_calibration='CALIBRATION_REQUIRED',semantic_provenance=observation.get('identity_source'))


class OfflineShadowConsumer:
    """Drain explicitly OFFLINE. Never installed or invoked by runtime hooks.

    All caches/results bounded. Missing events retire correlation context.
    Inputs are thawed private snapshots; existing issuer/reducer only, no old
    selector rerun. Failures are local diagnostics, never controller exceptions.
    """
    def __init__(self,*,limits,phase_config):
        from marvin_navigation_shadow import Accounting
        self.limits=limits;self.config=phase_config;self.records=deque(maxlen=limits.capacity)
        self.failures=0;self.last_index=0;self.mission=None;self.context=None
        self.accounting=Accounting();self.pending={};self.plans={};self.last_observation=None
        self.last_geometry=None;self.last_bridge=None;self.latest_wait=None
        self.retired_stamp=0;self.first_divergence=None
        self.completion_unresolved=False;self.terminal_failure=False
        self.detour_first_seen=None;self.detour_finished=None;self.recorded_veto_count=0
        self.accounting_incomplete=False
        self.completed_counts={'lateral':0,'pass':0,'alignment':0,'direct':0}

    def record(self,event,kind,**data):
        self.records.append(DiagnosticEvent(1,event.mission_id,EventType(kind),event.event_index,event.event_time,
            event.provenance,freeze(data,self.limits)))

    def failure(self,event,certificate,reason,missing=()):
        self.failures+=1
        try:self.record(event,EventType.CERTIFICATE_BUILD_FAILED,certificate_type=certificate,reason=str(reason),missing_facts=list(missing))
        except Exception:pass

    def limited_put(self,table,key,value):
        table[key]=value
        if len(table)>self.limits.capacity:table.pop(next(iter(table)))

    def process(self,event):
        from marvin_navigation_issuers import (IssuerContext,issue_observation,issue_geometry,issue_completion,
            issue_jit_veto,SnapshotCertificates,certificates_to_policy)
        from marvin_navigation_certificates import EvidenceKey
        from marvin_navigation_phases import Event,Phase,Side
        from marvin_navigation_policy import WatchdogEvidence
        from marvin_navigation_shadow import phase_policy,existing_policy_intent,compare,account,Accounting,PhaseResult
        from marvin_navigation_policy import NavigationIntent,IntentKind
        try:
            if type(event) is not DiagnosticEvent:raise ValueError('immutable event required')
            p=thaw(event.payload)
            if self.mission!=event.mission_id or event.event_type==EventType.MISSION_BEGIN:
                self.mission=event.mission_id;self.context=None;self.pending.clear();self.plans.clear();self.last_geometry=None;self.last_observation=None;self.accounting=Accounting();self.last_bridge=None
                self.completed_counts={'lateral':0,'pass':0,'alignment':0,'direct':0}
                self.retired_stamp=0;self.first_divergence=None;self.latest_wait=None
                self.completion_unresolved=False;self.terminal_failure=False
                self.detour_first_seen=None;self.detour_finished=None;self.recorded_veto_count=0
                self.accounting_incomplete=False
            if self.last_index and event.event_index!=self.last_index+1:
                self.pending.clear();self.plans.clear();self.last_geometry=None;self.last_observation=None
                # Preserve side commitment/retired-frame floor; losing an
                # event cannot create a new obstacle episode or action credit.
                self.accounting_incomplete=True
                self.failure(event,'EvidenceSnapshot','EVENT_GAP: no correlation across a lost producer boundary')
            self.last_index=event.event_index
            if event.event_type==EventType.MISSION_BEGIN:return
            if event.event_type==EventType.TERMINAL:
                self.record(event,EventType.TERMINAL,**p,native_diagnostic_elapsed=self.detour_elapsed(event.event_time));return
            if event.event_type==EventType.STOPPED_LIDAR_RECEIVED:self.latest_wait=p;return
            if event.event_type==EventType.BLOCKED_WAIT:
                self.latest_wait=p
                self.record(event,EventType.BLOCKED_WAIT,**p);return
            if event.event_type==EventType.ACTION_STOPPED:
                self.completion_unresolved=True
                from marvin_navigation_issuers import reject_delivery_contradictions
                try:reject_delivery_contradictions(p['result'])
                except Exception as exc:
                    self.terminal_failure=True;self.failure(event,'CompletionCertificate',exc)
                self.limited_put(self.pending,p['source_frame_stamp_ns'],p);self.last_bridge=p.get('bridge');return
            if event.event_type==EventType.ROUTE_REASSESSED:
                self.record(event,EventType.ROUTE_REASSESSED,**p)
                previous=self.pending.get(p.get('source_frame_stamp_ns'))
                if previous:previous.update(meaningful_progress=p.get('meaningful_progress'),meaningful_progress_reason=p.get('meaningful_progress_reason'))
                return
            session=p.get('producer_session');ctx=IssuerContext(event.mission_id,session,event.event_time,
                mission_owned=p.get('mission_owned'),producer_running=p.get('producer_running'))
            if event.event_type in (EventType.STRICT_OBSERVATION_ACCEPTED,EventType.OBSERVATION_RECORDED):
                self.last_observation=issue_observation(p['observation'],ctx,source_mission_id=event.mission_id)
                if not self.last_observation.ok:self.failure(event,'ObservationCertificate',self.last_observation.failure.reason,(self.last_observation.failure.field,))
                else:self.record(event,EventType.STRICT_OBSERVATION_ACCEPTED,certificate=asdict(self.last_observation.certificate),unsupported=list(self.last_observation.unsupported))
                # Preserve actual observation+associated raw scan to close the
                # post-STOP pairing gap, not a scan acquired solely for logging.
                pair=p.get('geometry_pair')
                if pair:
                    self.last_geometry=issue_geometry(pair[0],pair[1],{},ctx,source_mission_id=event.mission_id)
                    if not self.last_geometry.ok:
                        self.failure(event,'GeometryCertificate',self.last_geometry.failure.reason,(self.last_geometry.failure.field,))
                    else:self.record(event,EventType.GEOMETRY_EVALUATED,certificate=asdict(self.last_geometry.certificate),unsupported=list(self.last_geometry.unsupported),boundary='post_observation_association')
                if self.last_observation.ok and self.last_geometry and self.last_geometry.ok:
                    for stamp,completed in list(self.pending.items()):
                        if stamp>=self.last_observation.certificate.source_stamp_ns:continue
                        plan=self.plans.get(stamp)
                        if not plan:self.failure(event,'CompletionCertificate','Missing original planning snapshot');self.pending.pop(stamp);continue
                        jit=completed.get('jit_pair')
                        if not jit:self.failure(event,'CompletionCertificate','Missing original JIT geometry/freshness');self.pending.pop(stamp);continue
                        jit_key=EvidenceKey(event.mission_id,session,jit[0]['acquisition_sequence'],jit[0]['received_monotonic_seconds'],jit[0]['age_at_receipt_seconds'])
                        facts=native_completion_facts(completed)
                        c=issue_completion(facts,plan['geometry'].certificate.key,jit_key,self.last_geometry.certificate,self.last_observation.certificate,ctx,source_mission_id=event.mission_id)
                        if c.ok:
                            self.completion_unresolved=False
                            self.record(event,EventType.ACTION_COMPLETED,certificate=asdict(c.certificate),native_facts=facts,
                                provenance_note='Captured native acknowledgement/ledger/STOP; no displacement inferred',alignment_after=alignment_diagnostics(p['observation']),
                                alignment_before=alignment_diagnostics(plan['payload'].get('observation') or {}),
                                phase_at_planning=plan.get('shadow_phase'),
                                compatibility_avoidance_count=completed.get('avoidance_count'),
                                post_blocker_xy_m=self.last_geometry.certificate.blocker_xy_m,
                                semantic_gemini_source_followed=None if p['observation'].get('identity_source') is None else p['observation']['identity_source']=='gemini_marvin_candidate_selection',
                                reacquisition_attempts_before=plan['payload'].get('reacquisition_attempts'),
                                reacquisition_attempts_after=p.get('reacquisition_attempts'),
                                fresh_semantic_wait_recheck=p.get('fresh_semantic_wait_recheck'))
                            kind=c.certificate.primitive.value
                            count_kind='lateral' if kind.startswith('STRAFE_') else 'pass' if kind=='BYPASS_FORWARD' else 'alignment' if kind.startswith('TURN_') else 'direct'
                            self.completed_counts[count_kind]+=1
                            if self.context:
                                completion_data=certificates_to_policy(SnapshotCertificates(self.last_observation,self.last_geometry,completion=c),ctx,
                                    watchdog=WatchdogEvidence(prior_phase=self.context.phase,committed_side=self.context.committed_side),
                                    bridge=completed.get('bridge'),delivery_resolved=True,event=Event.ACTION_COMPLETED)
                                # Record actual completion, not counterfactual phase action credit.
                                self.accounting=account(self.accounting,completion_data,PhaseResult(self.context,NavigationIntent(IntentKind.STOP_REVERIFY,'completion_accounting_only')))
                        else:
                            self.terminal_failure=self.terminal_failure or c.failure.fail_closed
                            self.failure(event,'CompletionCertificate',c.failure.reason,(c.failure.field,))
                        self.pending.pop(stamp)
                return
            if event.event_type==EventType.GEOMETRY_EVALUATED:
                g=issue_geometry(p['lidar'],p['association'],p.get('selection') or {},ctx,source_mission_id=event.mission_id)
                self.last_geometry=g
                stamp=p['source_frame_stamp_ns']
                self.limited_put(self.plans,stamp,{'geometry':g,'payload':p})
                if not g.ok:self.failure(event,'GeometryCertificate',g.failure.reason,(g.failure.field,));return
                if g.certificate.direct_route_obstructed and self.detour_first_seen is None:
                    self.detour_first_seen=event.event_time
                if not g.certificate.direct_route_obstructed and (p.get('old_outcome') or {}).get('action_type')=='FORWARD':
                    self.detour_finished=event.event_time
                self.record(event,EventType.GEOMETRY_EVALUATED,certificate=asdict(g.certificate),unsupported=list(g.unsupported),boundary='planning')
                if self.last_observation is None or not self.last_observation.ok:
                    self.failure(event,'EvidenceSnapshot','Missing valid accepted observation');return
                if self.last_observation.certificate.source_stamp_ns!=stamp:
                    self.failure(event,'EvidenceSnapshot','Decision source stamp does not match accepted observation',('source_frame_stamp_ns',));return
                bundle=SnapshotCertificates(self.last_observation,g);nav_event=Event.OBSERVATION;old=p.get('old_outcome') or {};veto=None
            elif event.event_type==EventType.JIT_VETO_RECORDED:
                self.recorded_veto_count+=1
                stamp=p['source_frame_stamp_ns'];plan=self.plans.get(stamp)
                if not plan or not plan['geometry'].ok:self.failure(event,'JitVetoCertificate','Missing original planning geometry');return
                raw=p['result']['pre_transport_jit_veto'];g=issue_geometry(raw['lidar_snapshot'],raw['target_association'],p.get('selection') or {},ctx,source_mission_id=event.mission_id)
                result=dict(p['result']);ledger=p['source_stamp_consumed']
                if result.get('source_stamp_consumed') is not None and result['source_stamp_consumed'] is not ledger:
                    self.failure(event,'JitVetoCertificate','Native consumption contradicts actual ledger');return
                result['source_stamp_consumed']=ledger
                stationary=dict(retired_source_frame_stamp_ns=stamp if ledger else None,stop_result=p['stop_result'],bridge=p['bridge'],
                    active_forward=p['interlock'].get('active_forward'),pending_forward=p['interlock'].get('pending_forward'),
                    vetoed_primitive=plan['payload']['old_outcome']['action_type'],stationary_certified_monotonic_seconds=event.event_time)
                veto=issue_jit_veto(result,plan['geometry'].certificate.key,g.certificate,stationary,ctx,source_mission_id=event.mission_id)
                if not veto.ok:
                    self.terminal_failure=self.terminal_failure or veto.failure.fail_closed
                    self.failure(event,'JitVetoCertificate',veto.failure.reason,(veto.failure.field,));return
                planned_side=(plan['payload'].get('selection') or {}).get('direction')
                self.record(event,EventType.JIT_VETO_CERTIFIED,certificate=asdict(veto.certificate),
                    planned_side=planned_side,
                    planning_blocker_xy_m=plan['geometry'].certificate.blocker_xy_m,jit_blocker_xy_m=g.certificate.blocker_xy_m,
                    planning_lateral_separation_m=next((s.lateral_separation_m for s in plan['geometry'].certificate.sides if s.side.value==planned_side),None),
                    jit_lateral_separation_m=next((s.lateral_separation_m for s in g.certificate.sides if s.side.value==planned_side),None))
                self.retired_stamp=max(self.retired_stamp,stamp)
                if self.last_observation is None:return
                bundle=SnapshotCertificates(self.last_observation,g,veto=veto);nav_event=Event.ACTION_JIT_VETO;old={'action_type':'BLOCKED_WAIT','reason':result['reason']}
                self.last_bridge=p['bridge']
            else:return
            phase=self.context.phase if self.context else Phase.DIRECT;side=self.context.committed_side if self.context else None
            wd=WatchdogEvidence(avoidance_count=p.get('avoidance_count',0),prior_phase=phase,committed_side=side)
            self.last_observation.certificate.require_strict_current(**ctx.current(1.))
            data=certificates_to_policy(bundle,ctx,watchdog=wd,bridge=p['bridge'] if 'bridge' in p else self.last_bridge,
                delivery_resolved=False if self.terminal_failure else None if self.completion_unresolved else True if veto else p.get('delivery_resolved'),event=nav_event)
            chosen=phase_policy(data,self.config,context=self.context)
            if nav_event!=Event.ACTION_JIT_VETO and data.reference.frame_stamp_ns<=self.retired_stamp:
                chosen=PhaseResult(chosen.context,NavigationIntent(IntentKind.STOP_REVERIFY,'vetoed_source_authority_retired'))
            chosen=replace(chosen,context=replace(chosen.context,retired_stamp_ns=max(chosen.context.retired_stamp_ns,self.retired_stamp)))
            self.context=chosen.context;self.accounting=account(self.accounting,data,chosen)
            if stamp in self.plans:self.plans[stamp]['shadow_phase']=chosen.context.phase.value
            self.record(event,EventType.SHADOW_DECISION,phase=chosen.context.phase.value,phase_before=phase.value,intent=asdict(chosen.intent),
                calibration_required=chosen.calibration_required,accounting=asdict(self.accounting),
                completed_action_counts=dict(self.completed_counts),
                accounting_incomplete_due_to_event_loss=self.accounting_incomplete,
                phase_entered_at=chosen.context.phase_entered_at,detour_started_at=chosen.context.detour_started_at,
                legacy_avoidance_count=wd.avoidance_count,alignment=alignment_diagnostics(p.get('observation') or {}),
                native_wait=p.get('blocked_wait') or self.latest_wait,
                native_detour_started_at=self.detour_first_seen,native_detour_elapsed_seconds=self.detour_elapsed(event.event_time),
                recorded_zero_transport_veto_count=self.recorded_veto_count,
                equivalent_native_veto_count=None,
                stagnation_evidence='UNKNOWN_UNLESS_COMPARABLE_MEASURED_EVIDENCE',
                geometry=dict(sequence=g.certificate.key.sequence,blocker_xy_m=g.certificate.blocker_xy_m,
                    sides=[asdict(s) for s in g.certificate.sides],blocker_identity=None))
            existing=existing_policy_intent(old,data);comparison=compare(str(event.event_index),data,existing,chosen,provenance=event.provenance)
            if self.first_divergence is None and existing.kind!=chosen.intent.kind:self.first_divergence=event.event_index
            self.record(event,EventType.SHADOW_COMPARISON,old_intent=asdict(existing),shadow_intent=asdict(chosen.intent),same=comparison.old_intent.kind==comparison.phase_result.intent.kind,
                difference_category=comparison.category.value,reason=comparison.reason,evidence_completeness='SUPPORTED_CERTIFICATES_ONLY',calibration_required=chosen.calibration_required,
                first_counterfactual_divergence=self.first_divergence,
                path_semantics='INDEPENDENT_OBSERVED_SNAPSHOTS; NOT A COUNTERFACTUAL_TRAJECTORY')
        except Exception as e:self.failure(event,'Instrumentation',e)

    def detour_elapsed(self,now):
        """Observed epoch accounting only, not a watchdog or measured travel."""
        return None if self.detour_first_seen is None else max(0.,(self.detour_finished if self.detour_finished is not None else now)-self.detour_first_seen)

    def drain(self,buffer):
        try:
            for event in buffer.drain():self.process(event)
        except Exception:self.failures+=1

    def serialize_records(self):
        """Explicit offline operation, never part of producer capture."""
        try:return tuple(serialize_event(r) for r in self.records)
        except Exception:self.failures+=1;return ()
