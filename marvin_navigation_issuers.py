"""Pure, non-authoritative adapters for CURRENT production evidence.

No runtime instance, I/O, dispatch, authority ledger mutation, clock rebasing,
or perception algorithm is created here. Production remains authoritative.
Results are immutable certificates or explicit construction failures. Retained
JSON decimal stamps can be decoded exactly only through the explicit wire
adapter; canonical issuer inputs require integer stamps.
"""
import copy
from dataclasses import dataclass, fields
from enum import Enum
import math

from runtime import CognitiveRuntime
from lidar_perception import read_lidar_state, MAXIMUM_EFFECTIVE_AGE_SECONDS
from local_motion_safety_envelope import evaluate_local_motion_safety, LOCAL_LIDAR_PROTECTED_RADIUS_M
from marvin_blocked_wait import PRE_TRANSPORT_JIT_WAIT_REASONS, explicit_pre_transport_jit_veto
from marvin_navigation_certificates import (
    EvidenceKey, ObservationCertificate, GeometryCertificate, SideGeometry,
    CompletionCertificate, CompletionChronology, NativeBridgeZeroFrontier, JitVetoCertificate, DispatchedPrimitive,
)
from marvin_navigation_certificates import VisibilityClass
from marvin_navigation_phases import Phase, Side, Event, Primitive
from marvin_navigation_policy import (
    PolicyInput, EvidenceReference, TargetEvidence, HealthEvidence, SideEvidence,
    OutcomeEvidence, WatchdogEvidence, EvidenceHealth as H, Alignment, TargetVisibility,
)


class FailureCode(str, Enum):
    MISSING = 'UNSUPPORTED_MISSING_EVIDENCE'
    MALFORMED = 'MALFORMED_EVIDENCE'
    MISSION = 'CROSS_MISSION_EVIDENCE'
    SESSION = 'SESSION_OR_RESTART_MISMATCH'
    CHRONOLOGY = 'CHRONOLOGY_VIOLATION'
    STALE = 'STALE_EVIDENCE'
    TARGET = 'TARGET_AUTHORITY_INVALID'
    GEOMETRY = 'REQUIRED_GEOMETRY_INVALID'
    SAFETY = 'UNRESOLVED_SAFETY_OR_DELIVERY'


@dataclass(frozen=True, slots=True)
class ConstructionFailure:
    code: FailureCode
    field: str
    reason: str

    @property
    def fail_closed(self):
        return self.code in {FailureCode.MISSION, FailureCode.SESSION, FailureCode.SAFETY, FailureCode.MALFORMED}


@dataclass(frozen=True, slots=True)
class Fact:
    name: str
    value: str | int | float | bool | None
    origin: str

    def __post_init__(self):
        if type(self.name) is not str or type(self.origin) is not str or type(self.value) not in (str,int,float,bool,type(None)):
            raise ValueError('Only immutable scalar evidence facts; no handles/callables')
        if type(self.value) is float and not math.isfinite(self.value): raise ValueError('finite fact')


_CERTS=(ObservationCertificate,GeometryCertificate,CompletionCertificate,JitVetoCertificate)


@dataclass(frozen=True, slots=True)
class Issuance:
    certificate: ObservationCertificate | GeometryCertificate | CompletionCertificate | JitVetoCertificate | None
    failure: ConstructionFailure | None
    facts: tuple = ()
    unsupported: tuple = ()

    def __post_init__(self):
        if (self.certificate is None)==(self.failure is None): raise ValueError('Exactly one success/failure')
        if self.certificate is not None and type(self.certificate) not in _CERTS: raise ValueError('certificate type')
        if self.failure is not None and type(self.failure) is not ConstructionFailure: raise ValueError('failure type')
        if type(self.facts) is not tuple or any(type(x) is not Fact for x in self.facts): raise ValueError('frozen facts')
        if type(self.unsupported) is not tuple or any(type(x) is not str for x in self.unsupported): raise ValueError('frozen gaps')

    @property
    def ok(self): return self.certificate is not None

    def fact(self,name):
        return next((f.value for f in self.facts if f.name==name),None)


@dataclass(frozen=True, slots=True)
class IssuerContext:
    mission_id: str
    producer_session: str
    now: float
    camera_floor_ns: int = 0
    lidar_floor: int = 0
    mission_owned: bool | None = None
    producer_running: bool | None = None

    def __post_init__(self):
        if any(type(x) is not str or not x for x in (self.mission_id,self.producer_session)): raise ValueError('independent owner/session context required')
        if type(self.now) not in (int,float) or not math.isfinite(self.now) or self.now<0: raise ValueError('monotonic epoch')
        for x in (self.camera_floor_ns,self.lidar_floor):
            if type(x) is not int or x<0: raise ValueError('exact floors')
        for x in (self.mission_owned,self.producer_running):
            if x is not None and type(x) is not bool: raise ValueError('explicit context health')

    def current(self,age):
        return dict(mission_id=self.mission_id,session=self.producer_session,now=self.now,max_age=age)


class EvidenceError(ValueError):
    def __init__(self,code,field,reason):self.failure=ConstructionFailure(code,field,reason)


def require(test,code,field,reason):
    if not test: raise EvidenceError(code,field,reason)


def required(row,key,typ=None):
    value=row.get(key)
    require(value is not None,FailureCode.MISSING,key,'No recorded/produced value; never default to zero/false')
    if typ is not None:require(type(value) is typ,FailureCode.MALFORMED,key,'Exact type required')
    return value


def numeric(row,key):
    x=required(row,key)
    require(type(x) in (int,float) and math.isfinite(x),FailureCode.MALFORMED,key,'Finite numeric evidence required')
    return x


def binding(ctx,source_mission,record=None):
    require(type(source_mission) is str and bool(source_mission),FailureCode.MISSING,'mission_id','Missing source mission binding')
    require(source_mission==ctx.mission_id,FailureCode.MISSION,'mission_id','Source mission differs from independent context')
    if record is not None:
        require(required(record,'producer_session',str)==ctx.producer_session,FailureCode.SESSION,'producer_session','Sensor owner differs; restart boundary is terminal')
        seq=required(record,'acquisition_sequence',int)
        require(seq>0 and seq>=ctx.lidar_floor,FailureCode.CHRONOLOGY,'acquisition_sequence','Sequence regressed or malformed')


def capture(function):
    """Issuer failures are data. Do not swallow exceptions from production I/O."""
    def adapted(*args,**kwargs):
        try:return function(*args,**kwargs)
        except EvidenceError as e:return Issuance(None,e.failure)
        except (ValueError,TypeError,KeyError) as e:
            return Issuance(None,ConstructionFailure(FailureCode.MALFORMED,'schema',str(e)))
    return adapted


def decode_retained_stamps(value):
    """Exact JSON-wire decoding, not float normalization or stamp consumption."""
    stamp_names={'source_frame_stamp_ns','post_action_source_frame_stamp_ns','identity_source_frame_stamp_ns','tracker_source_frame_stamp_ns'}
    def visit(v):
        if isinstance(v,dict):
            out={}
            for k,x in v.items():
                if k in stamp_names and type(x) is str:
                    require(x.isascii() and x.isdecimal() and str(int(x))==x,FailureCode.MALFORMED,k,'Noncanonical decimal stamp')
                    out[k]=int(x)
                else:out[k]=visit(x)
            return out
        if isinstance(v,list):return [visit(x) for x in v]
        return copy.deepcopy(v)
    return visit(value)


@capture
def issue_observation(observation,ctx,*,source_mission_id):
    binding(ctx,source_mission_id)
    o=copy.deepcopy(observation);t=required(o,'opencv_tracker',dict)
    stamp=required(o,'source_frame_stamp_ns',int)
    require(stamp==required(t,'source_frame_stamp_ns',int),FailureCode.CHRONOLOGY,'source_frame_stamp_ns','Observation and tracker stamp disagree')
    require(stamp>ctx.camera_floor_ns,FailureCode.CHRONOLOGY,'camera_floor','New strict observation must exceed retired floor')
    if o.get('post_action_source_frame_stamp_ns') is not None:
        floor=required(o,'post_action_source_frame_stamp_ns',int)
        require(stamp>floor>=0,FailureCode.CHRONOLOGY,'post_action_camera_floor','Recorded camera must exceed the exact producer floor')
    receipt=numeric(t,'received_monotonic_seconds')
    require(0<=ctx.now-receipt<=CognitiveRuntime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS,FailureCode.STALE,'observation_age','Current production camera receipt freshness exceeded')
    active=required(t,'active',bool);matched=required(t,'matched',bool);identity=required(o,'identity_confirmed',bool)
    require(active and matched and identity,FailureCode.TARGET,'identity_tracker','Existing tracker/identity verdict unhealthy')
    quality=numeric(t,'quality');threshold=numeric(t,'threshold')
    require(quality>=max(.8,threshold),FailureCode.TARGET,'tracker_quality','Existing range/tracker quality semantics reject target')
    arrival=required(o,'arrival',dict);binding(ctx,source_mission_id,arrival)
    trusted=required(arrival,'target_range_association_trusted',bool)
    arrived=required(arrival,'arrived_at_marvin',bool)
    controller=required(o,'controller',dict)
    box=required(t,'bbox',dict);bbox=tuple(numeric(box,k) for k in ('x1','y1','x2','y2'))
    width=required(t,'image_width',int);height=required(t,'image_height',int)
    require(0<=bbox[0]<bbox[2]<=width and 0<=bbox[1]<bbox[3]<=height,FailureCode.MALFORMED,'bbox','Malformed/clipped image bounds')
    # Reuse the exact production current-frame predicate. ARRIVED is evidence
    # only and deliberately cannot pass the production motion-action predicate.
    strict=CognitiveRuntime._marvin_v2_action_observation(o,t,controller.get('state'),controller.get('decision'))
    require(strict is not None or (arrived and trusted),FailureCode.TARGET,'strict_observation','Production V2 current-frame predicate rejected observation')
    vis=o.get('visibility_classification')
    # No numeric detour loss threshold exists. Accept only explicitly issued
    # typed classifications; missing remains CALIBRATION_REQUIRED.
    require(vis is None or type(vis) is VisibilityClass,FailureCode.MALFORMED,'visibility','No uncalibrated string/numeric classifier')
    visibility=vis or VisibilityClass.CALIBRATION_REQUIRED
    distance=arrival.get('verified_marvin_distance_m')
    if trusted:require(type(distance) in (int,float) and math.isfinite(distance),FailureCode.MISSING,'verified_range','Trusted range needs its actual distance')
    # Camera age is defined by runtime ONLY from local receipt. Zero here is
    # the existing clock-model origin, not a claim of zero source delivery age.
    c=ObservationCertificate(EvidenceKey(ctx.mission_id,ctx.producer_session,stamp,receipt,0.),stamp,
        identity,matched,quality,threshold,width,bbox,visibility,trusted,distance,arrived)
    c.require_strict_current(newer_than_stamp=ctx.camera_floor_ns or None,**ctx.current(1.))
    error=t.get('horizontal_error');tolerance=controller.get('center_tolerance_pixels')
    facts=(Fact('camera_age_basis','runtime_local_receipt_only','runtime._marvin_motion_stamp_is_fresh'),
        Fact('tracker_active',active,'opencv_tracker'),Fact('image_height',height,'opencv_tracker'),
        Fact('horizontal_error',error,'opencv_tracker'),Fact('centering_tolerance',tolerance,'controller'),
        Fact('controller_decision',controller.get('decision'),'controller'),
        Fact('observation_sequence',o.get('observation_sequence'),'production observation'),
        Fact('observation_key_sequence_basis','exact_source_stamp_order_not_camera_counter','runtime source-frame contract'),
        Fact('post_action_source_frame_stamp_ns',o.get('post_action_source_frame_stamp_ns'),'strict observation'))
    gaps=('numeric_visibility_calibration',) if vis is None else ()
    if o.get('observation_sequence') is None:gaps+=('separate_observation_counter',)
    return Issuance(c,None,facts,gaps)


@capture
def issue_geometry(lidar,association,selection,ctx,*,source_mission_id,lateral_probe_durations=None):
    """Wrap existing evaluated route/bypass outputs; reuse production safety.

    No route or bypass projection is recomputed by this adapter. Existing
    primitive safety is called with explicit now and recorded probe duration;
    it owns point-cloud, footprint and sector calculations. Missing planner
    alternatives stay unknown, rather than becoming negative feasibility.
    """
    binding(ctx,source_mission_id,lidar);binding(ctx,source_mission_id,association)
    seq=lidar['acquisition_sequence']
    require(association['acquisition_sequence']==seq,FailureCode.CHRONOLOGY,'association_sequence','Association is not from this scan')
    # The reader must receive actual freshness inputs, never an invented zero.
    numeric(lidar,'received_monotonic_seconds');numeric(lidar,'age_at_receipt_seconds')
    state=read_lidar_state(lidar,expected_session=ctx.producer_session,now=ctx.now)
    require(state.get('valid') is True and state.get('available') is True,
        FailureCode.STALE if state.get('reason') in ('stale','invalid_freshness') else FailureCode.GEOMETRY,'lidar',str(state.get('reason')))
    geometry=required(state,'local_motion_geometry',dict)
    require(geometry.get('valid') is True,FailureCode.GEOMETRY,'local_motion_geometry','Invalid producer geometry')
    # Production rotation envelope requires all octants. Inspect its validated
    # geometry output instead of recreating the required-sector algorithm.
    coverage=evaluate_local_motion_safety(state,expected_session=ctx.producer_session,angular_z=.25,duration=.5,now=ctx.now)
    require(coverage.get('geometry') is not None,FailureCode.GEOMETRY,'required_sectors',str(coverage.get('reason')))
    route=required(association,'route',dict)
    require(route.get('valid') is True,FailureCode.GEOMETRY,'route','Current production route is invalid')
    required(route,'route_to_marvin_obstructed',bool);required(route,'route_occupancy',int)
    numeric(route,'corridor_overlap_m')
    if selection:
        binding(ctx,source_mission_id,selection)
        require(selection['acquisition_sequence']==seq,FailureCode.CHRONOLOGY,'selection_sequence','Planner output belongs to another scan')
    sides=[];facts=[];unsupported=[]
    candidates=association.get('local_bypass_candidates') or {}
    for side in Side:
        option=((selection or {}).get('options') or {}).get('STRAFE_'+side.value) or {}
        lateral=option.get('hard_safety_permitted')
        if lateral is not None:
            require(type(lateral) is bool,FailureCode.MALFORMED,'lateral_feasible','Boolean planner verdict required')
            duration=numeric(option,'requested_duration')
            # Revalidate exact producer/parameters using the unchanged envelope.
            evaluated=evaluate_local_motion_safety(state,expected_session=ctx.producer_session,
                linear_y=.08 if side==Side.LEFT else -.08,duration=duration,now=ctx.now,lateral_swept_footprint=True)
            require(evaluated['permitted']==lateral,FailureCode.GEOMETRY,'lateral_verdict','Recorded hard gate contradicts current production calculation')
        elif lateral_probe_durations and side.value in lateral_probe_durations:
            # Worker-only revalidation of the exact preceding native planner
            # probe duration, on this stopped scan. No selector is rerun and
            # no prior feasibility verdict is carried across scan frontiers.
            duration=lateral_probe_durations[side.value]
            require(type(duration) in (int,float) and math.isfinite(duration) and duration>0,
                FailureCode.MALFORMED,'lateral_probe_duration','Recorded positive planner duration required')
            evaluated=evaluate_local_motion_safety(state,expected_session=ctx.producer_session,
                linear_y=.08 if side==Side.LEFT else -.08,duration=duration,now=ctx.now,lateral_swept_footprint=True)
            lateral=evaluated['permitted']
            facts.append(Fact(side.value+'.lateral_probe_duration',duration,'preceding native planner duration; stopped-scan production safety revalidation'))
        else:unsupported.append(side.value+'.lateral_feasible')
        b=candidates.get(side.value) or {}
        if b:
            binding(ctx,source_mission_id,b)
            require(b['acquisition_sequence']==seq,FailureCode.CHRONOLOGY,'bypass_sequence','Recomputed target belongs to another scan')
        x=b.get('local_bypass_target_x_m');y=b.get('local_bypass_target_y_m')
        require((x is None)==(y is None),FailureCode.MALFORMED,'target','Partly missing target')
        target=None if x is None else (x,y)
        blocker_y=route.get('blocking_obstacle_y_m')
        signed=None if blocker_y is None else (-blocker_y if side==Side.LEFT else blocker_y)
        separation=signed if signed is not None and signed>=0 else None
        minimum=route.get('blocking_obstacle_minimum_side_separation_m') if separation is not None else None
        # Existing schema cannot represent a blocker on the opposite side.
        # Preserve signed diagnostic fact; missing established separation is
        # never changed into zero or another side's positive separation.
        if signed is not None and signed<0:unsupported.append(side.value+'.opposite_side_separation')
        forward=b.get('forward_safety') or {}
        capsule=forward.get('permitted')
        for name in ('bypass_forward_permitted','route_to_bypass_obstructed'):
            if b.get(name) is not None:
                require(type(b[name]) is bool,FailureCode.MALFORMED,name,'Exact producer boolean required')
        if capsule is not None:
            require(forward.get('protected_radius_m')==LOCAL_LIDAR_PROTECTED_RADIUS_M,FailureCode.GEOMETRY,'protected_radius','Production capsule radius changed')
        prediction=(((selection or {}).get('options') or {}).get('BYPASS_FORWARD') or {}).get('route_progress') or {}
        gain=prediction.get('bypass_longitudinal_progress_m') if (selection or {}).get('direction')==side.value else None
        sides.append(SideGeometry(side,lateral,separation,(selection or {}).get(side.value.lower()+'_clearance_m'),
            target,seq if target is not None else None,b.get('bypass_corridor_occupancy'),b.get('bypass_corridor_overlap_m'),
            b.get('route_to_bypass_obstructed'),capsule,gain,minimum,separation,
            'evaluate_marvin_route.blocking_obstacle_minimum_side_separation_m; signed blocker side'))
        facts.extend((Fact(side.value+'.pass_permitted',b.get('bypass_forward_permitted'),'plan_local_bypass'),
                      Fact(side.value+'.minimum_side_separation_m',minimum,'evaluate_marvin_route.blocking_obstacle_minimum_side_separation_m'),
                      Fact(side.value+'.blocker_center_separation_m',separation,'plan_local_bypass: -side_sign * blocking_obstacle_y_m'),
                      Fact(side.value+'.signed_separation_m',signed,'evaluate_marvin_route output'),
                      Fact(side.value+'.prediction_is_measured',False,'existing planner prediction')))
    bx=route.get('blocking_obstacle_x_m');by=route.get('blocking_obstacle_y_m')
    require((bx is None)==(by is None),FailureCode.MALFORMED,'blocker_xy','Partly missing blocker')
    key=EvidenceKey(ctx.mission_id,ctx.producer_session,seq,numeric(state,'received_monotonic_seconds'),numeric(state,'age_at_receipt_seconds'))
    c=GeometryCertificate(key,True,True,True,route.get('route_to_marvin_obstructed'),route.get('route_occupancy'),
        route.get('corridor_overlap_m'),None if bx is None else (bx,by),tuple(sides))
    c.require_current(**ctx.current(MAXIMUM_EFFECTIVE_AGE_SECONDS))
    return Issuance(c,None,tuple(facts),tuple(unsupported))


def bridge_stationary(bridge):
    return (isinstance(bridge,dict) and bridge.get('ok') is True and bridge.get('status')=='READY'
        and bridge.get('ros_ready') is True and isinstance(bridge.get('motion'),dict)
        and all(type(bridge['motion'].get(k)) in (int,float) and bridge['motion'][k]==0 for k in ('linear_x','linear_y','angular_z'))
        and bridge['motion'].get('streaming') is False)


def chronology(*,used_stamp,observation_stamp,planning_key,jit_key,post_key,post_camera,stop_time,stop_floor):
    require(type(used_stamp) is int and used_stamp==observation_stamp,FailureCode.CHRONOLOGY,'used_stamp','Action must use exact observation N')
    require(jit_key.sequence>=planning_key.sequence and post_key.sequence>stop_floor>=jit_key.sequence,
        FailureCode.CHRONOLOGY,'lidar_order','Planning≤JIT≤STOP floor<new stopped scan')
    require(post_camera.source_stamp_ns>used_stamp,FailureCode.CHRONOLOGY,'post_camera','Post-action camera must exceed N')
    require(min(post_key.received_monotonic_seconds,post_camera.key.received_monotonic_seconds)>stop_time>=jit_key.received_monotonic_seconds,
        FailureCode.CHRONOLOGY,'STOP_order','Stopped evidence must follow confirmed STOP')


def reject_delivery_contradictions(record):
    """A summary success cannot override an explicit nested executor failure."""
    stack=[(record,0)];seen=set()
    while stack:
        value,depth=stack.pop()
        require(depth<=30,FailureCode.MALFORMED,'result_depth','Malformed nested result')
        if isinstance(value,(dict,list)):
            if id(value) in seen:continue
            seen.add(id(value))
        if isinstance(value,dict):
            for name,item in value.items():
                if name in ('delivery_uncertain','delivery_uncertainty','interrupted'):
                    require(item is False,FailureCode.SAFETY,name,'Uncertain/interrupted nested executor evidence')
                if name in ('transport_error','error','exception'):
                    require(not item,FailureCode.SAFETY,name,'Explicit executor failure')
                stack.append((item,depth+1))
        elif isinstance(value,list):stack.extend((x,depth+1) for x in value)


@capture
def issue_completion(record,planning_key,jit_key,post_geometry,post_observation,ctx,*,source_mission_id):
    binding(ctx,source_mission_id)
    r=copy.deepcopy(record)
    used=required(r,'source_frame_stamp_ns',int);auth=required(r,'observation_source_frame_stamp_ns',int)
    primitive=DispatchedPrimitive(required(r,'action_type',str))
    # Exact producer facts, not inferred from the command or its duration.
    names=('execution_authorized','transport_attempted','transport_confirmed','delivery_uncertain',
           'motion_executed','full_step_completed','automatic_stop','source_stamp_consumed')
    vals={n:required(r,n,bool) for n in names}
    reject_delivery_contradictions(r)
    facts=tuple(Fact(n,vals[n],'production action result') for n in names)
    require(vals['execution_authorized'] and vals['transport_attempted'] and vals['transport_confirmed']
        and vals['motion_executed'] and vals['full_step_completed'] and vals['source_stamp_consumed']
        and not vals['delivery_uncertain'],FailureCode.SAFETY,'completion','No complete certain physical dispatch')
    stop=r.get('stop_result')
    require(stop is None or (type(stop) is dict and stop.get('ok') is True),FailureCode.SAFETY,'stop_result','Explicit STOP failed or malformed')
    require(vals['automatic_stop'] or (stop or {}).get('ok') is True,FailureCode.SAFETY,'STOP','No automatic/explicit STOP confirmation')
    retained_bridge=r.get('native_bridge_zero_frontier')
    if retained_bridge is not None:
        require(type(retained_bridge) is NativeBridgeZeroFrontier
            and retained_bridge.mission_id==ctx.mission_id and retained_bridge.source_stamp_ns==used,
            FailureCode.MISSION,'Bridge_frontier','Native verification must bind this mission and dispatch')
    else:
        require(bridge_stationary(required(r,'bridge_after_stop',dict)),FailureCode.SAFETY,'Bridge','READY/ROS/x/y/yaw/streaming proof failed')
    require(type(post_geometry) is GeometryCertificate and type(post_observation) is ObservationCertificate,
        FailureCode.MISSING,'post_certificates','No valid matched stopped geometry/strict observation')
    for key in (planning_key,jit_key,post_geometry.key,post_observation.key):
        require(type(key) is EvidenceKey,FailureCode.MALFORMED,'key','Immutable exact evidence key required')
        require(key.mission_id==ctx.mission_id,FailureCode.MISSION,'mission_id','Cross-mission completion')
        require(key.producer_session==ctx.producer_session,FailureCode.SESSION,'session','Restart/session crossing')
    stop_time=numeric(r,'stopped_monotonic_seconds');floor=required(r,'stop_lidar_sequence',int)
    frontier=r.get('chronology')
    frontier=CompletionChronology(**frontier) if frontier is not None else None
    if retained_bridge is not None:
        require(frontier is not None and retained_bridge.verified_monotonic_seconds==frontier.bridge_zero_confirmed_monotonic_seconds,
            FailureCode.CHRONOLOGY,'Bridge_frontier','Retained verification must match the distinct Bridge poll frontier')
    if frontier is not None:
        require(vals['automatic_stop'] and frontier.automatic_stop_completed_monotonic_seconds==stop_time,
            FailureCode.CHRONOLOGY,'STOP_contract','Automatic STOP frontier must match bounded acknowledgement')
        require(frontier.bridge_zero_confirmed_monotonic_seconds<=ctx.now,FailureCode.CHRONOLOGY,
            'Bridge_frontier','Bridge stationary confirmation must already exist')
    chronology(used_stamp=used,observation_stamp=auth,planning_key=planning_key,jit_key=jit_key,
        post_key=post_geometry.key,post_camera=post_observation,stop_time=stop_time,stop_floor=floor)
    post_geometry.require_current(**ctx.current(MAXIMUM_EFFECTIVE_AGE_SECONDS))
    post_observation.require_strict_current(newer_than_stamp=max(used,ctx.camera_floor_ns),**ctx.current(1.))
    progress=r.get('meaningful_progress');gain=r.get('measured_lateral_gain_m')
    c=CompletionCertificate(primitive,used,planning_key,jit_key,post_geometry,post_observation,True,True,True,True,False,
        progress,stop_time,True,floor,gain,frontier)
    c.require_completed(mission_id=ctx.mission_id,session=ctx.producer_session)
    return Issuance(c,None,facts+(Fact('meaningful_progress_reason',r.get('meaningful_progress_reason'),'production measured progress'),))


@capture
def issue_retained_completion(row,primitive,planning_key,jit_key,post_geometry,post_observation,ctx,*,source_mission_id):
    """Explicit offline adapter for the native progress-diagnostic contract.

    No missing result dictionary or Bridge response is invented. Dispatch,
    transport and automatic STOP come from the retained command callback;
    completion/consumption from action_result; stationary verification from
    its separately retained native frontier. Decimal stamps must be decoded
    by the caller's explicit wire adapter before entering this canonical API.
    """
    binding(ctx,source_mission_id)
    camera=required(row,'authorizing_camera',dict);stamp=required(camera,'source_frame_stamp_ns',int)
    command=required(row,'command',dict);ack=required(command,'bridge_acknowledgement',dict)
    reject_delivery_contradictions(row)
    require(ack.get('ok') is True and ack.get('mode')=='bounded' and ack.get('automatic_stop') is True
        and ack.get('returned_immediately') is False,FailureCode.SAFETY,'bounded_ack','No native automatic STOP contract')
    native_primitive=row.get('action_type')
    if native_primitive is None:
        yaw=command.get('angular_z')
        native_primitive='TURN_LEFT' if type(yaw) in (int,float) and yaw>0 else 'TURN_RIGHT' if type(yaw) in (int,float) and yaw<0 else None
    require(native_primitive==primitive,FailureCode.CHRONOLOGY,'action_primitive','Primitive must match the retained native dispatch')
    for axis in ('linear_x','linear_y','angular_z','duration'):
        value=numeric(command,axis)
        require(type(ack.get(axis)) in (int,float) and ack[axis]==value,
            FailureCode.SAFETY,'bounded_ack','Acknowledgement contradicts native command '+axis)
    if primitive in ('FORWARD','BYPASS_FORWARD'):
        from behavior_manager import BehaviorManager
        require(BehaviorManager._is_canonical_marvin_bounded_forward_result(ack,speed=command['linear_x'],duration=command['duration']),
            FailureCode.SAFETY,'bounded_ack','Noncanonical bounded forward acknowledgement')
    start=numeric(command,'start_monotonic_seconds');ack_time=numeric(command,'completion_monotonic_seconds')
    zero=numeric(row,'bridge_zero_verified_monotonic_seconds')
    r=dict(action_type=primitive,source_frame_stamp_ns=stamp,observation_source_frame_stamp_ns=stamp,
        execution_authorized=True,transport_attempted=True,transport_confirmed=True,delivery_uncertain=False,
        motion_executed=required(row,'motion_executed',bool),full_step_completed=required(row,'full_step_completed',bool),
        automatic_stop=True,source_stamp_consumed=required(row,'source_stamp_consumed',bool),
        stopped_monotonic_seconds=ack_time,stop_lidar_sequence=jit_key.sequence,
        native_bridge_zero_frontier=NativeBridgeZeroFrontier(ctx.mission_id,stamp,zero),
        chronology=dict(transport_call_started_monotonic_seconds=start,
            dispatch_lower_bound_monotonic_seconds=max(start,jit_key.received_monotonic_seconds),
            transport_acknowledged_monotonic_seconds=ack_time,
            automatic_stop_completed_monotonic_seconds=ack_time,bridge_zero_confirmed_monotonic_seconds=zero))
    result=issue_completion(r,planning_key,jit_key,post_geometry,post_observation,ctx,source_mission_id=source_mission_id)
    if not result.ok:return result
    return Issuance(result.certificate,None,result.facts+(Fact('retained_completion_basis',
        'native command dispatch/ack, successful action result and Bridge verification frontier',
        'MarvinProgressDiagnostics; not reconstructed raw responses'),))


@capture
def issue_jit_veto(result,planning_key,geometry,stationary,ctx,*,source_mission_id):
    binding(ctx,source_mission_id)
    require(ctx.mission_owned is True and ctx.producer_running is True,FailureCode.SAFETY,'owner_producer','Current production ownership/worker must be proven')
    require(isinstance(result,dict) and explicit_pre_transport_jit_veto(result),FailureCode.SAFETY,'pre_transport_jit_veto','Existing production zero-transport predicate rejects ambiguous contract')
    stamp=required(result,'source_frame_stamp_ns',int)
    require(required(result,'source_stamp_consumed',bool) is True,FailureCode.MISSING,'retired_authority','Do not infer consumed authority from planning/veto reason')
    require(required(stationary,'retired_source_frame_stamp_ns',int)==stamp,FailureCode.CHRONOLOGY,'retired_stamp','Runtime ledger and action disagree')
    require((stationary.get('stop_result') or {}).get('ok') is True,FailureCode.SAFETY,'STOP','STOP not confirmed')
    require(bridge_stationary(stationary.get('bridge')),FailureCode.SAFETY,'Bridge','Stationary READY/ROS proof missing')
    require(stationary.get('active_forward') is False and stationary.get('pending_forward') is False,FailureCode.SAFETY,'interlock','Forward active/pending or unknown')
    require(type(geometry) is GeometryCertificate,FailureCode.MISSING,'JIT_geometry','No valid fresh complete geometry certificate')
    raw=result['pre_transport_jit_veto']['lidar_snapshot'];binding(ctx,source_mission_id,raw)
    require(raw['acquisition_sequence']==geometry.key.sequence,FailureCode.CHRONOLOGY,'JIT_sequence','Negative dispatch evidence is not this certified geometry')
    require(type(planning_key) is EvidenceKey and planning_key.mission_id==ctx.mission_id,FailureCode.MISSION,'planning_mission','Cross-mission plan')
    require(planning_key.producer_session==ctx.producer_session,FailureCode.SESSION,'planning_session','Cross-session plan')
    c=JitVetoCertificate(DispatchedPrimitive(required(stationary,'vetoed_primitive',str)),stamp,planning_key,geometry,
        result['reason'],True,False,True,True,True,False,False,numeric(stationary,'stationary_certified_monotonic_seconds'))
    c.require_safe_replan(allowlisted_reasons=PRE_TRANSPORT_JIT_WAIT_REASONS,**ctx.current(MAXIMUM_EFFECTIVE_AGE_SECONDS))
    return Issuance(c,None,(Fact('transport_attempted',False,'explicit_pre_transport_jit_veto'),
        Fact('physical_dispatch_confirmed',False,'explicit_pre_transport_jit_veto')))


@dataclass(frozen=True, slots=True)
class SnapshotCertificates:
    observation: Issuance
    geometry: Issuance
    completion: Issuance | None = None
    veto: Issuance | None = None

    def __post_init__(self):
        if any(type(x) is not Issuance for x in (self.observation,self.geometry)):
            raise ValueError('Immutable typed issuances required')
        if any(x is not None and type(x) is not Issuance for x in (self.completion,self.veto)):
            raise ValueError('Optional immutable typed issuances required')


def build_snapshot(snapshot,ctx):
    """Pure normalized boundary dictionary, sourced from retained runtime data.

    Keys reference producer objects, not clients: mission_id, observation,
    lidar, association, selection; optional completion, planning_key, jit_key,
    stopped_geometry/observation, veto_result and stationary evidence. Missing
    inputs produce construction failures. Never mutate input or consumption set.
    """
    s=copy.deepcopy(snapshot);mission=s.get('mission_id')
    o=issue_observation(s.get('observation') or {},ctx,source_mission_id=mission)
    g=issue_geometry(s.get('lidar') or {},s.get('association') or {},s.get('selection') or {},ctx,source_mission_id=mission)
    completion=None;veto=None
    if 'completion' in s:
        completion=issue_completion(s['completion'],s.get('planning_key'),s.get('jit_key'),
            s.get('stopped_geometry'),s.get('stopped_observation'),ctx,source_mission_id=mission)
    if 'veto_result' in s:
        veto=issue_jit_veto(s['veto_result'],s.get('planning_key'),g.certificate,s.get('stationary') or {},ctx,source_mission_id=mission)
    return SnapshotCertificates(o,g,completion,veto)


def build_runtime_snapshot(runtime_record,ctx):
    """Accept a retained /status dictionary, without querying /status.

    Runtime diagnostics often intentionally trim clouds and completion facts.
    Those exports are not certificates. Missing facts stay construction
    failures; live instrumentation must capture the original producer objects
    before trimming. Independent context supplies ownership/session, never a
    self-certifying health default from the record being inspected.
    """
    try:
        r=decode_retained_stamps(runtime_record.get('runtime',runtime_record))
    except EvidenceError as e:
        failed=Issuance(None,e.failure)
        return SnapshotCertificates(failed,failed)
    d=r.get('marvin_progress_diagnostics') or {}
    o=d.get('last_observation') or {}
    return build_snapshot({'mission_id':d.get('mission_id'), 'observation':o,
        'lidar':d.get('last_lidar') or {},
        'association':d.get('last_target_association') or {},
        'selection':o.get('local_avoidance_selection') or {}},ctx)


@capture
def issue_native_completion(result,*,action_type,authorizing_stamp,planning_key,
        jit_key,post_geometry,post_observation,ctx,source_mission_id,
        stopped_monotonic_seconds,stop_lidar_sequence):
    """Map native action results, not requested command distance.

    Explicit attempt/uncertainty facts are mandatory. Current normal mission
    exports do not always retain them. Their absence must not be converted to
    False even when an acknowledgement says OK. This deliberately exposes a
    producer/export gap for the future instrumentation slice.
    """
    r=copy.deepcopy(result)
    reject_delivery_contradictions(r)
    ack=(r.get('lateral_step') or {}).get('lateral_result') or (r.get('approach_result') or {}).get('forward_result') or r.get('turn_result') or {}
    require(ack.get('ok') is not False,FailureCode.SAFETY,'transport_ack','Negative transport acknowledgement contradicts completion')
    native={k:r.get(k) for k in ('source_frame_stamp_ns','execution_authorized',
        'transport_attempted','transport_confirmed','delivery_uncertain','motion_executed',
        'full_step_completed','source_stamp_consumed','bridge_after_stop','stop_result',
        'meaningful_progress','meaningful_progress_reason','measured_lateral_gain_m')}
    # Automatic STOP is the actual Bridge acknowledgement, not a duration.
    native.update(action_type=action_type,observation_source_frame_stamp_ns=authorizing_stamp,
        automatic_stop=ack.get('automatic_stop'),stopped_monotonic_seconds=stopped_monotonic_seconds,
        stop_lidar_sequence=stop_lidar_sequence)
    return issue_completion(native,planning_key,jit_key,post_geometry,post_observation,ctx,
        source_mission_id=source_mission_id)


def certificates_to_policy(bundle,ctx,*,watchdog: WatchdogEvidence,bridge=None,delivery_resolved=None,event=Event.OBSERVATION):
    """No health assumptions: absent ownership/Bridge/etc remain UNKNOWN."""
    def health(x):return H.OK if x is True else H.BAD if x is False else H.UNKNOWN
    fatal=any(i and i.failure and i.failure.fail_closed for i in (bundle.observation,bundle.geometry,bundle.completion,bundle.veto))
    o=bundle.observation.certificate;g=bundle.geometry.certificate
    sides={s.side:s for s in g.sides} if g else {}
    def projected(side):
        v=sides.get(side)
        if v is None:return SideEvidence(side)
        permitted=bundle.geometry.fact(side.value+'.pass_permitted')
        return SideEvidence(side,v.lateral_feasible,v.minimum_side_separation_m,permitted,v.pass_occupancy,v.pass_overlap_m,
            v.protected_capsule_clear,v.target_recomputed_sequence==g.key.sequence if v.target_xy_m is not None else None)
    error=bundle.observation.fact('horizontal_error');tol=bundle.observation.fact('centering_tolerance')
    alignment=Alignment.UNKNOWN
    if type(error) in (int,float) and type(tol) in (int,float):
        alignment=Alignment.CENTERED if abs(error)<=tol else Alignment.RIGHT if error>0 else Alignment.LEFT
    visibility=TargetVisibility.CALIBRATION_REQUIRED
    if o and o.visibility==VisibilityClass.CENTERING:visibility=TargetVisibility.TRACK_OK
    if o and o.visibility==VisibilityClass.TARGET_LOST:visibility=TargetVisibility.LOST
    if o and o.visibility==VisibilityClass.VISIBILITY_MARGIN_LOW:
        visibility=TargetVisibility.MAINTAIN_RIGHT if error is not None and error>0 else TargetVisibility.MAINTAIN_LEFT if error is not None and error<0 else TargetVisibility.CALIBRATION_REQUIRED
    stopped=None if bridge is None else bridge_stationary(bridge)
    complete=bundle.completion.certificate if bundle.completion else None
    if bridge is None and event==Event.ACTION_COMPLETED and complete is not None:
        # Historical native completion already includes its independently
        # verified stationary frontier. Do not fabricate a raw Bridge response.
        stopped=complete.bridge_zero
    veto=bundle.veto.certificate if bundle.veto else None
    outcome=OutcomeEvidence()
    if complete:
        kind=Primitive.TURN if complete.primitive in (DispatchedPrimitive.TURN_LEFT,DispatchedPrimitive.TURN_RIGHT) else Primitive(complete.primitive.value)
        outcome=OutcomeEvidence(last_completed_primitive=kind,completion_confirmed=True,
            recorded_meaningful_route_progress=complete.measured_direct_route_progress,event_frame_stamp_ns=complete.used_source_stamp_ns)
    if veto:
        outcome=OutcomeEvidence(event_frame_stamp_ns=veto.retired_source_stamp_ns,zero_transport_certified=True,stationary_stop_confirmed=True,old_authority_retired=True)
        stopped=True;delivery_resolved=True
    required_ok=o is not None and g is not None
    committed=sides.get(watchdog.committed_side)
    if (committed is not None and g.direct_route_obstructed and committed.minimum_side_separation_m is not None
            and committed.minimum_side_separation_m<.15 and committed.lateral_feasible is None):
        # Lost pass separation is real, but missing same-scan repair feasibility
        # cannot invent a strafe. Existing policy returns STOP_REVERIFY.
        required_ok=False
    if event==Event.ACTION_JIT_VETO:required_ok=required_ok and veto is not None
    if event==Event.ACTION_COMPLETED:required_ok=required_ok and complete is not None
    # A valid but incomplete side inventory cannot claim a planner dead end.
    if g and g.direct_route_obstructed and any(s.lateral_feasible is None for s in g.sides):required_ok=False
    h=HealthEvidence(health(ctx.mission_owned),H.BAD if fatal else health(ctx.producer_running),health(g is not None),
        health(g.required_sectors_valid if g else None),health(stopped),health(delivery_resolved),H.OK if required_ok else H.UNKNOWN)
    target=TargetEvidence(health(o is not None),health(o is not None),o.trusted_standoff if o else False,
        bool(o and g and not g.direct_route_obstructed and o.target_range_trusted),alignment,visibility,error,o.tracker_quality if o else None)
    return PolicyInput(EvidenceReference(ctx.now,ctx.producer_session,g.key.sequence if g else None,o.source_stamp_ns if o else None,required_ok),
        target,h,projected(Side.LEFT),projected(Side.RIGHT),g.direct_route_obstructed if g else None,
        g.route_occupancy if g else None,g.route_overlap_m if g else None,
        g.blocker_xy_m[0] if g and g.blocker_xy_m else None,g.blocker_xy_m[1] if g and g.blocker_xy_m else None,
        watchdog.committed_side,outcome,watchdog,event,'CERTIFICATE_BUILT_NONAUTHORITATIVE')
