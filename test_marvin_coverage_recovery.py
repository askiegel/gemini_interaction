"""Offline coverage/strict identity regression; no camera, provider or robot IO."""
import copy
import hashlib
import json
import math
import pathlib
import time
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from behavior_manager import BehaviorManager
from marvin_coverage_candidates import marvin_coverage_candidates
from marvin_local_tracker import MarvinLocalTracker
from semantic_vision import JpegFrame, SemanticVisionClient

F=pathlib.Path(__file__).parent/'test_fixtures'
FIXTURE=json.loads((F/'marvin_coverage_recovery.json').read_text())
JPEG=(F/'marvin_coverage_recovery.jpg').read_bytes()
TARGET=FIXTURE['manual_annotation']['bbox']
TV=FIXTURE['television_bbox']
SELECTED=FIXTURE['expected_coverage_selection']['index']


def coverage(box, target=TARGET):
    intersection=max(0,min(box['x2'],target['x2'])-max(box['x1'],target['x1']))*max(0,min(box['y2'],target['y2'])-max(box['y1'],target['y1']))
    return intersection/((target['x2']-target['x1'])*(target['y2']-target['y1']))


class NoRobot:
    def __getattr__(self,name):
        raise AssertionError('Offline observation touched robot: '+name)


class Responses:
    def __init__(self,primary=None,recovery=None):
        self.models=self;self.requests=[]
        self.primary=primary or FIXTURE['normal_response']['parsed']
        self.recovery=FIXTURE['coverage_response']['parsed'] if recovery is None else recovery
    def generate_content(self,**kwargs):
        self.requests.append(kwargs)
        is_coverage='fixed camera coverage crops' in kwargs['contents'][0]
        return SimpleNamespace(parsed=copy.deepcopy(self.recovery if is_coverage else self.primary))


class Harness:
    """Real retained pixels; subsequent frame stamps/receipts are synthetic."""
    def __init__(self,*,primary=None,recovery=None,stamp_delta=1,quality=None):
        from vision_adapter import VisionAdapter
        self.provider=Responses(primary,recovery)
        self.semantic=SemanticVisionClient(client=self.provider,model='gemini-2.5-flash')
        self.seed=None;self.frame_count=0;self.quality=quality;self.stamp_delta=stamp_delta
        self.created=[]
        self.semantic.fetch_frame=self.frame
        vision=VisionAdapter.__new__(VisionAdapter);vision.last_payload=FIXTURE['generic_detections']
        values=iter(copy.deepcopy(FIXTURE['proposal_frames']))
        vision.fetch_detection_proposals=lambda:next(values)
        self.manager=BehaviorManager(robot_client=NoRobot(),vision_adapter=vision,semantic_vision=self.semantic)
        self.manager.target_lock=SimpleNamespace(target_label='marvin',snapshot=lambda:{'tracking_mode':'UNLOCKED'})
        self.manager.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS=.04
        self.manager.MARVIN_POST_TURN_FRAME_POLL_SECONDS=.001
        def factory(frame,bbox):
            assert self.manager._marvin_v2_tracker_episode is None
            assert not hasattr(self.manager,'trusted_range')
            tracker=MarvinLocalTracker(frame,bbox);self.created.append((frame,dict(bbox),tracker))
            if quality is not None:
                native=tracker.update
                def update(f):
                    bbox=native(f);tracker.last_quality=quality;return bbox
                tracker.update=update
            return tracker
        self.manager.marvin_local_tracker_factory=factory
    def frame(self):
        self.frame_count+=1
        base=FIXTURE['frame']['source_frame_stamp_ns']
        # New pixels are not claimed: this is a synthetic stationary future frame.
        stamp=base if self.frame_count==1 else base+(self.frame_count-1)*self.stamp_delta
        now=datetime.now(timezone.utc)+timedelta(microseconds=self.frame_count)
        f=JpegFrame(JPEG,640,480,now.isoformat(),stamp,time.monotonic())
        if self.seed is None:self.seed=f
        return f
    def acquire(self):
        return self.manager._acquire_marvin_proposal_tracker_observation(require_fresh_gemini=True)


@pytest.mark.parametrize('width,height',[(640,480),(320,240),(1280,720),(480,640)])
def test_deterministic_bounded_complete_image_coverage(width,height):
    a=marvin_coverage_candidates(width,height,123)
    assert a==marvin_coverage_candidates(width,height,123)
    assert a==[dict(c,source_frame_stamp_ns=123) for c in marvin_coverage_candidates(width,height,456)]
    assert len(a)==16
    mask=np.zeros((height,width),bool)
    for c in a:
        b=c['bbox'];MarvinLocalTracker._validate_bbox(b,width,height)
        mask[b['y1']:b['y2'],b['x1']:b['x2']]=True
        assert b!={'x1':0,'y1':0,'x2':width,'y2':height}
    assert mask.all()
    assert a[0]['bbox']['x1']==a[0]['bbox']['y1']==0
    assert a[3]['bbox']['x2']==width and a[3]['bbox']['y2']==height
    assert a[0]['bbox']['x2']>a[1]['bbox']['x1']
    assert a[0]['bbox']['y2']>a[2]['bbox']['y1']


@pytest.mark.parametrize('width,height',[(640,480),(320,240),(1280,720),(480,640)])
@pytest.mark.parametrize('position',[(0,0),(1,0),(0,1),(1,1),(.5,0),(.5,1),(0,.5),(1,.5),(.5,.5)])
def test_equivalent_target_size_is_covered_at_edges_corners_and_center(width,height,position):
    tw=int(round(width*73/640));th=int(round(height*108/480))
    x=int(round((width-tw)*position[0]));y=int(round((height-th)*position[1]))
    target={'x1':x,'y1':y,'x2':x+tw,'y2':y+th}
    assert max(coverage(c['bbox'],target) for c in marvin_coverage_candidates(width,height,1))>=.9


@pytest.mark.parametrize('args',[(0,480,1),(640,0,1),(63,480,1),(640,480,True),(640,480,1.0),(640,480,-1)])
def test_invalid_coverage_input_fails_closed(args):
    with pytest.raises(ValueError):marvin_coverage_candidates(*args)


def test_exact_retained_frame_and_primary_proposal_gap():
    assert hashlib.sha256(JPEG).hexdigest()==FIXTURE['jpeg_sha256']
    h=Harness();c,status,d=h.manager._confirm_marvin_proposal_candidates_with_status()
    g=h.manager._filter_marvin_proposal_geometry(c,d)
    assert len(c)==23 and len(g)==21 and len(g[:8])==8
    assert all(coverage(x['bbox'])==0 for x in c)


def test_retained_coverage_candidate_is_small_robot_crop_excluding_television():
    c=marvin_coverage_candidates(640,480,FIXTURE['frame']['source_frame_stamp_ns'])[SELECTED]
    assert c['bbox']=={'x1':334,'y1':168,'x2':434,'y2':312}
    assert coverage(c['bbox'])>=.9 and coverage(c['bbox'],TV)==0
    # Complete head and central body; only a small annotation margin is outside.
    assert coverage(c['bbox'],{'x1':340,'y1':206,'x2':381,'y2':248})==1
    assert coverage(c['bbox'],{'x1':338,'y1':248,'x2':385,'y2':308})==1
    seed=BehaviorManager._expand_marvin_tracker_seed_bbox(c['bbox'],640,480)
    assert seed=={'x1':314,'y1':161,'x2':454,'y2':319}
    assert coverage(seed)==1
    tracker=MarvinLocalTracker(Harness().frame(),seed)
    assert tracker.template.shape==(158,140)


def test_numbered_sheet_and_existing_candidate_schema_map_to_original_boxes():
    h=Harness();frame=h.frame();c=marvin_coverage_candidates(640,480,frame.source_frame_stamp_ns)
    r=h.semantic.select_marvin_coverage_candidate(frame,c)
    k=h.provider.requests[0]
    assert set(k['config'].response_schema['properties'])=={'target','confirmed','candidate_index'}
    assert 'bbox' not in k['config'].response_schema['properties']
    assert hashlib.sha256(k['contents'][1].inline_data.data).hexdigest()==FIXTURE['contact_sheet_sha256']
    assert r['candidate_index']==SELECTED and c[r['candidate_index']]['bbox']==FIXTURE['expected_coverage_selection']['bbox']
    assert k['config'].http_options.retry_options.attempts==1
    assert k['config'].http_options.timeout==12000


def test_successful_primary_does_not_construct_coverage_or_call_fallback(monkeypatch):
    h=Harness(primary={'target':'marvin','confirmed':True,'candidate_index':0})
    def denied(*a,**k):raise AssertionError('coverage called despite primary success')
    monkeypatch.setattr('behavior_manager.marvin_coverage_candidates',denied)
    monkeypatch.setattr(h.semantic,'select_marvin_coverage_candidate',denied)
    result=h.acquire()
    assert result['identity_confirmed'] is True
    assert len(h.provider.requests)==1
    assert result['geometry_source']=='yolo_proposal'
    assert result['identity_selection']['identity_selection_path']=='PROPOSAL'
    assert result['identity_selection']['coverage_recovery_attempted'] is False


def test_retained_pipeline_uses_same_frame_once_and_real_tracker_on_synthetic_newer_frames():
    h=Harness();result=h.acquire();selection=result['identity_selection']
    assert len(h.provider.requests)==2
    assert hashlib.sha256(h.provider.requests[0]['contents'][1].inline_data.data).hexdigest()==FIXTURE['normal_contact_sheet_sha256']
    assert selection['normal_proposal_count']==23 and selection['normal_submitted_crop_count']==8
    assert selection['coverage_candidate_count']==16 and selection['coverage_selected_index']==SELECTED
    assert selection['provisional_tracker_initialized'] is True
    assert h.created[0][0] is h.seed
    assert result['identity_source_frame_stamp_ns']==h.seed.source_frame_stamp_ns
    assert result['opencv_tracker']['source_frame_stamp_ns']>h.seed.source_frame_stamp_ns
    assert result['opencv_tracker']['quality']>=.8 and result['opencv_tracker']['threshold']==.8
    assert BehaviorManager._marvin_v2_preview_is_verified(result)
    assert result['geometry_source']=='deterministic_coverage' and result['yolo_seed_bbox'] is None
    assert 'route' not in result and 'trusted_range' not in result
    assert h.manager.mark_strict_v2_action_dispatched(result['source_frame_stamp_ns'],'turn')


@pytest.mark.parametrize('index',[True,1.5,None,'10',-1,16,100])
def test_invalid_selected_index_never_seeds_tracker(index):
    h=Harness(recovery={'target':'marvin','confirmed':True,'candidate_index':index})
    r=h.acquire()
    assert r['found'] is False and not h.created
    assert h.manager._marvin_v2_tracker_episode is None and len(h.provider.requests)==2


@pytest.mark.parametrize('response',[
    {'target':'marvin','confirmed':False,'candidate_index':-1},
    {'target':'marvin','confirmed':False},
    {'target':'marvin','confirmed':True},
    {'target':'tv','confirmed':True,'candidate_index':1},
    {'target':'marvin','confirmed':False,'candidate_index':1},
    {'target':'marvin','confirmed':'true','candidate_index':10},
])
def test_absence_malformed_or_television_claim_fails_closed(response):
    h=Harness(recovery=response);result=h.acquire()
    assert result['found'] is False and not h.created
    assert result.get('identity_confirmed') is not True
    assert not result.get('motion_authorized_marvin_candidate',False)
    assert 'route' not in result and 'trusted_range' not in result
    assert h.manager._marvin_v2_tracker_episode is None
    assert len(h.provider.requests)==2


@pytest.mark.parametrize('delta',[0,-1])
def test_duplicate_or_older_tracker_stamp_clears_provisional_state(delta):
    h=Harness(stamp_delta=delta);result=h.acquire()
    assert result['found'] is False and len(h.created)==1
    assert h.manager._marvin_v2_tracker_episode is None
    assert not result.get('motion_authorized_marvin_candidate',False)


@pytest.mark.parametrize('quality',[.799,float('nan'),None,False])
def test_unverified_tracker_quality_cannot_promote_identity(quality):
    h=Harness(quality=quality)
    if quality is None:
        # Force inconsistent provider diagnostics rather than native match quality.
        native=h.manager.marvin_local_tracker_factory
        def factory(*a):
            t=native(*a);t.preview_diagnostics=lambda:{'quality':None,'threshold':.8};return t
        h.manager.marvin_local_tracker_factory=factory
    r=h.acquire();assert r['found'] is False
    assert h.manager._marvin_v2_tracker_episode is None
    assert not r.get('motion_authorized_marvin_candidate',False)


def test_selection_alone_is_not_verified_motion_or_range_evidence():
    h=Harness();frame=h.frame();c=marvin_coverage_candidates(640,480,frame.source_frame_stamp_ns)
    selected=h.semantic.select_marvin_coverage_candidate(frame,c)
    assert not BehaviorManager._marvin_v2_preview_is_verified(selected)
    assert not h.manager._marvin_motion_authorized_candidate(c[SELECTED],selected,require_fresh_gemini=True,
        identity_source='gemini_marvin_candidate_selection',identity_source_frame_stamp_ns=frame.source_frame_stamp_ns)
    assert h.manager._marvin_v2_tracker_episode is None
    assert not {'identity_confirmed','route','trusted_range'}&selected.keys()


def test_recovery_provider_error_has_no_retry_or_provisional_authority():
    h=Harness()
    native=h.provider.generate_content
    def failing(**kwargs):
        if 'fixed camera coverage crops' in kwargs['contents'][0]:
            h.provider.requests.append(kwargs)
            raise TimeoutError('offline provider timeout')
        return native(**kwargs)
    h.provider.generate_content=failing
    result=h.acquire()
    assert result['found'] is False and not h.created
    assert len(h.provider.requests)==2
    assert h.manager._marvin_v2_tracker_episode is None
    assert result['identity_selection']['coverage_recovery_attempted'] is True


def test_recovery_preemption_clears_provisional_episode():
    from behavior_manager import _SemanticPreempted
    h=Harness()
    def preempt(*args,**kwargs):
        h.manager._marvin_v2_tracker_episode={'provisional':True}
        raise _SemanticPreempted()
    h.manager._acquire_strict_v2_tracker_observation_from_candidate=preempt
    with pytest.raises(_SemanticPreempted):h.acquire()
    assert h.manager._marvin_v2_tracker_episode is None
    assert len(h.provider.requests)==2


def test_existing_association_does_not_reseed_unrelated_tracker_same_observation():
    h=Harness()
    h.manager._marvin_v2_tracker_episode={'marvin_tracker':object(),'tracker_bbox':{'x1':0,'y1':0,'x2':100,'y2':100},'last_tracker_source_frame_stamp_ns':1}
    r=h.acquire()
    assert r['found'] is False and r['reason']=='marvin_v2_semantic_tracker_association_failed'
    assert not h.created and h.manager._marvin_v2_tracker_episode is None


def test_read_only_runtime_replay_reaches_normal_range_and_route_without_motion():
    from test_find_marvin_runtime import _v2_runtime, _v2_preview
    from test_marvin_lidar_standoff import lidar_at
    h=Harness();runtime,_=_v2_runtime(_v2_preview())
    runtime.behavior_manager=h.manager
    lidar=lidar_at(1.4463040047729645)
    # Synthetic bounded surface centered at the native tracker's projected bearing.
    distance=1.4463040047729645;bearing=math.atan((320-384)/320)
    lidar['local_motion_geometry']['points'][-7:]=[
        {'x_m':distance*math.cos(bearing),'y_m':distance*math.sin(bearing)+(n-3)/1000} for n in range(7)]
    runtime.world_model.get_lidar_obstacles.side_effect=None
    runtime.world_model.get_lidar_obstacles.return_value=lidar
    assert runtime._marvin_alignment_consumed_source_frame_stamps==set()
    result=runtime.observe_find_marvin_v2()
    assert result['read_only'] and result['executed'] is False
    assert result['identity_confirmed'] and result['target_found']
    assert result['target_range_association_trusted'] is True
    assert result['arrival']['route']['valid'] is True
    assert runtime._marvin_alignment_consumed_source_frame_stamps==set()
    assert result['identity_selection']['identity_selection_path']=='COVERAGE_RECOVERY'
    assert 'image_bytes' not in json.dumps(result)


def test_failed_runtime_observation_has_no_range_or_route_authority():
    from test_find_marvin_runtime import _v2_runtime, _v2_preview
    h=Harness(recovery={'target':'marvin','confirmed':False,'candidate_index':-1})
    runtime,_=_v2_runtime(_v2_preview());runtime.behavior_manager=h.manager
    result=runtime.observe_find_marvin_v2()
    assert result['identity_confirmed'] is False and result['target_found'] is False
    assert 'arrival' not in result and 'route' not in result
    assert runtime._marvin_alignment_consumed_source_frame_stamps==set()
    assert runtime._marvin_alignment_observation is None


def test_navigation_sources_are_byte_identical_to_validated_baseline():
    expected=FIXTURE['navigation_hashes']
    for file,digest in expected.items():
        assert hashlib.sha256((F.parent/file).read_bytes()).hexdigest()==digest
    assert MarvinLocalTracker.MIN_MATCH_QUALITY==.8
