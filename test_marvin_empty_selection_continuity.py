"""Offline retained-metadata replay and real OpenCV continuation experiments.

No real Gemini/camera/robot IO. Neighboring JPEGs are retained live captures;
future frames/receipts in continuity tests are explicitly simulated.
"""
import copy
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from behavior_manager import BehaviorManager, _SemanticPreempted
from marvin_local_tracker import MarvinLocalTracker
from runtime import CognitiveRuntime
from semantic_vision import JpegFrame, SemanticVisionClient
from test_find_marvin_closed_loop import delayed_runtime, make_runtime, motions, run

FIXTURES = Path(__file__).parent / 'test_fixtures'
RETAINED = json.loads((FIXTURES / 'marvin_4756378f_empty_selection.json').read_text())
PIXELS = [(FIXTURES / f'marvin_4756378f_neighbor-{i}.jpg').read_bytes() for i in range(2)]
BOX = {'x1': 270, 'y1': 103, 'x2': 398, 'y2': 326}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline experiment attempted network/service IO')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)


class NoRobot:
    def __getattr__(self, name):
        raise AssertionError('Perception touched robot: ' + name)


class Harness:
    def __init__(self, monkeypatch):
        self.clock = [100.]
        monkeypatch.setattr('behavior_manager.time.monotonic', lambda: self.clock[0])
        monkeypatch.setattr('behavior_manager.time.sleep', lambda n: self.advance(max(.001, n)))
        self.stamp = 1000000000
        self.calls, self.frames, self.trackers = [], [], []
        self.response = {'target': 'marvin', 'confirmed': True, 'candidate_index': 0}
        self.frame_mode = 'fresh'
        self.quality = None
        self.shift = 0
        self.semantic_delay = .1
        self.on_semantic = None
        self.on_update = None
        semantic = SemanticVisionClient(client=SimpleNamespace(models=self), model='offline')
        semantic.fetch_frame = self.frame
        self.manager = BehaviorManager(robot_client=NoRobot(), semantic_vision=semantic)
        self.manager.target_lock = SimpleNamespace(target_label='marvin', snapshot=lambda: {})
        self.manager._confirm_marvin_proposal_candidates_with_status = lambda **_kw: (
            [{'bbox': dict(BOX), 'image_width': 640, 'image_height': 480,
              'proposal_label': 'dog', 'source_frame_stamp_ns': self.stamp}], 'target_confirmed', {})
        self.manager.marvin_local_tracker_factory = self.factory

    def advance(self, seconds):
        self.clock[0] += seconds

    def frame(self):
        self.advance(.01)
        if self.frame_mode != 'cached':
            self.stamp += 10000000
        receipt = self.clock[0] - (10 if self.frame_mode == 'stale_receipt' else 0)
        pixels = PIXELS[len(self.frames) % 2]
        if self.frame_mode == 'corrupt':
            pixels = b'not-a-jpeg'
        f = JpegFrame(pixels, 640, 480,
            datetime.fromtimestamp(self.clock[0], timezone.utc).isoformat(), self.stamp, receipt)
        self.frames.append(f)
        return f

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        self.advance(self.semantic_delay)
        if self.on_semantic:
            self.on_semantic()
        if isinstance(self.response, Exception):
            raise self.response
        return SimpleNamespace(parsed=copy.deepcopy(self.response), text=None)

    def factory(self, frame, bbox):
        tracker = MarvinLocalTracker(frame, bbox)
        self.trackers.append(tracker)
        native = tracker.update
        def update(fresh):
            box = native(fresh)
            if self.on_update:
                self.on_update()
            if self.quality is not None:
                tracker.last_quality = self.quality
                if self.quality < .80:
                    return None
            if box and self.shift:
                box = dict(box, x1=box['x1'] + self.shift, x2=box['x2'] + self.shift)
            return box
        tracker.update = update
        return tracker

    def observe(self):
        return self.manager.observe_find_marvin_v2()['preview_result']

    def seed(self):
        preview = self.observe()
        assert BehaviorManager._marvin_v2_preview_is_verified(preview)
        return preview


def test_empty_response_is_unknown_not_explicit_nondetection(monkeypatch):
    h = Harness(monkeypatch)
    h.response = None
    frame = h.frame()
    with pytest.raises(ValueError, match='^marvin_candidate_selection_response_empty$'):
        h.manager.semantic_vision.select_marvin_candidate(frame, [{'bbox': BOX}])
    h.response = {'target': 'marvin', 'confirmed': False, 'candidate_index': -1}
    assert h.manager.semantic_vision.select_marvin_candidate(frame, [{'bbox': BOX}])['confirmed'] is False
    assert len(h.calls) == 2


def test_recent_episode_survives_empty_response_on_two_real_opencv_matches(monkeypatch):
    h = Harness(monkeypatch)
    seed = h.seed()
    episode = h.manager._marvin_v2_tracker_episode
    h.manager.prepare_find_marvin_tracker_handoff()
    assert h.manager._marvin_v2_tracker_episode is episode
    h.response = None
    preview = h.observe()
    assert BehaviorManager._marvin_v2_preview_is_verified(preview)
    assert preview['identity_source'] == 'marvin_locked_tracker_continuity'
    assert preview['identity_source_frame_stamp_ns'] == seed['identity_source_frame_stamp_ns']
    assert preview['opencv_tracker']['source_frame_stamp_ns'] > h.frames[-5].source_frame_stamp_ns
    assert preview['semantic_gap_tracker_continuity']['empty_response_count'] == 1
    assert preview['semantic_gap_tracker_continuity']['minimum_tracker_iou'] >= .7
    assert preview['strict_tracker_episode']['initialized_this_observation'] is False
    assert preview['opencv_tracker']['quality'] >= .8
    assert len(h.trackers) == 1 and len(h.calls) == 2
    assert h.manager._marvin_v2_tracker_episode['marvin_tracker'] is episode['marvin_tracker']


@pytest.mark.parametrize('failure', ['cached', 'stale_receipt', 'corrupt', 'low_quality',
                                    'identity_switch', 'too_old', 'slow_semantic', 'cleared'])
def test_empty_response_cannot_bridge_failed_or_stale_continuity(monkeypatch, failure):
    h = Harness(monkeypatch); h.seed(); h.response = None
    if failure in {'cached', 'stale_receipt', 'corrupt'}:
        # The semantic input is valid; break only the post-response frames.
        h.on_semantic = lambda: setattr(h, 'frame_mode', failure)
    elif failure == 'low_quality':
        h.quality = .799
    elif failure == 'identity_switch':
        h.shift = 80
    elif failure == 'too_old':
        h.advance(3.001)
    elif failure == 'slow_semantic':
        h.semantic_delay = 3.001
    elif failure == 'cleared':
        h.on_update = h.manager._clear_marvin_v2_tracker_episode
    preview = h.observe()
    assert not BehaviorManager._marvin_v2_preview_is_verified(preview)
    assert preview['motion_authorized_marvin_candidate'] is False
    assert preview['reason'] == 'marvin_empty_selection_reacquisition_required'
    assert h.manager._marvin_v2_tracker_episode is None
    assert len(h.trackers) == 1


def test_repeated_empty_response_requires_semantic_reacquisition(monkeypatch):
    h = Harness(monkeypatch); h.seed(); h.response = None
    assert BehaviorManager._marvin_v2_preview_is_verified(h.observe())
    rejected = h.observe()
    assert rejected['target_found'] is False
    assert rejected['reason'] == 'marvin_empty_selection_reacquisition_required'
    assert h.manager._marvin_v2_tracker_episode is None
    assert len(h.calls) == 3 and len(h.trackers) == 1
    h.response = {'target': 'marvin', 'confirmed': True, 'candidate_index': 0}
    assert BehaviorManager._marvin_v2_preview_is_verified(h.observe())
    assert len(h.trackers) == 2
    assert h.manager._marvin_v2_tracker_episode['semantic_gap_used'] is False


def test_post_action_tracking_does_not_renew_an_empty_response_gap(monkeypatch):
    h = Harness(monkeypatch); h.seed(); h.response = None
    first = h.observe()
    stamp = first['opencv_tracker']['source_frame_stamp_ns']
    assert h.manager.mark_strict_v2_action_dispatched(stamp, 'offline-simulated-forward')
    # No physical command: exercise just the perception side of post-action.
    rejected = h.observe()
    assert rejected['reason'] == 'marvin_empty_selection_reacquisition_required'
    assert h.manager._marvin_v2_tracker_episode is None
    assert len(h.calls) == 3


@pytest.mark.parametrize('response', [ValueError('marvin_candidate_selection_response_empty:max_tokens'),
    ValueError('marvin_candidate_selection_response_empty:safety'),
    ValueError('marvin_candidate_selection_response_invalid'), OSError('offline transport')])
def test_other_provider_failures_never_use_gap(monkeypatch, response):
    h = Harness(monkeypatch); h.seed(); h.response = response
    preview = h.observe()
    assert not BehaviorManager._marvin_v2_preview_is_verified(preview)
    assert h.manager._marvin_v2_tracker_episode is None


def test_semantic_identity_switch_clears_episode(monkeypatch):
    h = Harness(monkeypatch); h.seed()
    h.manager._confirm_marvin_proposal_candidates_with_status = lambda **_kw: (
        [{'bbox': dict(x1=10, y1=30, x2=110, y2=200), 'image_width': 640,
          'image_height': 480, 'source_frame_stamp_ns': h.stamp}], 'target_confirmed', {})
    switched = h.observe()
    assert switched['reason'] == 'marvin_v2_semantic_tracker_association_failed'
    assert h.manager._marvin_v2_tracker_episode is None


def test_explicit_negative_never_uses_the_empty_response_bridge(monkeypatch):
    h = Harness(monkeypatch); h.seed()
    h.response = {'target': 'marvin', 'confirmed': False, 'candidate_index': -1}
    negative = h.observe()
    assert negative['target_found'] is False
    assert negative['reason'] == 'marvin_identity_not_confirmed'
    assert h.manager._marvin_v2_tracker_episode is None


@pytest.mark.parametrize('field,value', [
    ('identity_source', 'other_identity'),
    ('identity_source_frame_stamp_ns', None),
    ('marvin_tracker', None),
    ('last_tracker_received_monotonic_seconds', float('nan')),
    ('last_tracker_received_monotonic_seconds', 10000.),
])
def test_old_confirmation_without_valid_episode_cannot_authorize(monkeypatch, field, value):
    h = Harness(monkeypatch); h.seed(); h.response = None
    h.manager._marvin_v2_tracker_episode[field] = value
    preview = h.observe()
    assert not BehaviorManager._marvin_v2_preview_is_verified(preview)
    assert h.manager._marvin_v2_tracker_episode is None


def test_successful_semantic_recheck_resets_gap_budget_without_reseeding(monkeypatch):
    h = Harness(monkeypatch); h.seed(); h.response = None
    assert BehaviorManager._marvin_v2_preview_is_verified(h.observe())
    h.response = {'target': 'marvin', 'confirmed': True, 'candidate_index': 0}
    assert BehaviorManager._marvin_v2_preview_is_verified(h.observe())
    assert h.manager._marvin_v2_tracker_episode['semantic_gap_used'] is False
    h.response = None
    assert BehaviorManager._marvin_v2_preview_is_verified(h.observe())
    assert len(h.trackers) == 1


def test_stop_during_gap_frame_refresh_preempts_without_resurrection(monkeypatch):
    h = Harness(monkeypatch); h.seed(); h.response = None
    stopped = [False]
    def guard():
        if stopped[0]:
            raise _SemanticPreempted()
    h.manager.execution_authorization_provider = lambda: not stopped[0]
    h.manager._strict_v2_current_execution_guard = lambda: guard
    # Enter the acquisition core with its actual mission guard.
    h.on_update = lambda: stopped.__setitem__(0, True)
    with pytest.raises(_SemanticPreempted):
        h.manager._acquire_marvin_proposal_tracker_observation(
            require_fresh_gemini=True, execution_guard=guard)
    assert h.manager._marvin_v2_tracker_episode is None


def test_retained_failed_observation_replays_stopped_without_motion(tmp_path, monkeypatch):
    r, b, _, events, _ = make_runtime(tmp_path, monkeypatch, [])
    original = copy.deepcopy(RETAINED['final_observation'])
    # The Runtime's public observation normalization returns these same fields.
    r._observe_find_marvin_v2 = lambda **_kw: copy.deepcopy(original)
    result = run(r)
    assert result['reason'] == RETAINED['expected_original_reason']
    assert result['actions_executed'] == 0 and motions(events) == []
    assert result['safe_approach_complete'] is False and events[-1] == 'stop'


def test_retained_preflight_hiatus_requires_reacquisition_not_old_identity(tmp_path, monkeypatch):
    r, b, _, events, clock = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .8)])
    b.observe_find_marvin_v2()  # Fresh simulated Gemini and tracker episode.
    old = b._marvin_v2_tracker_episode
    preflight = RETAINED['preflight_observation']
    semantic = RETAINED['last_semantic_frame']
    hiatus = (int(semantic['source_frame_stamp_ns']) - int(preflight['source_frame_stamp_ns'])) / 1e9
    assert hiatus > b.TARGET_MAX_AGE_SECONDS
    clock[0] += int(hiatus * 1e9)
    b.semantic_vision.select_marvin_candidate = lambda *_a: (_ for _ in ()).throw(
        ValueError('marvin_candidate_selection_response_empty'))
    result = run(r)
    assert result['reason'] == 'marvin_empty_selection_reacquisition_required'
    assert b._marvin_v2_tracker_episode is None and old['semantic_gap_used'] is False
    assert result['actions_executed'] == 0 and motions(events) == []
    assert events[-1] == 'stop'


@pytest.mark.parametrize('fault', [None, 'collision', 'stop', 'stale'])
@pytest.mark.parametrize('error', [0, 120])
def test_runtime_gap_uses_ordinary_dispatch_stop_and_collision_guards(tmp_path, monkeypatch, fault, error):
    r, b, robot, events, clock = delayed_runtime(tmp_path, monkeypatch,
        [(error, .8), (error, .8), (0, .5)])
    seed = b.observe_find_marvin_v2()
    def empty(frame, candidates):
        b.semantic_calls += 1
        if fault == 'stop':
            r.submit_intent({'intent': 'STOP'})
        raise ValueError('marvin_candidate_selection_response_empty')
    b.semantic_vision.select_marvin_candidate = empty
    if fault == 'collision':
        b.unsafe_forward = True
    if fault == 'stale':
        original = b.frame
        def stale():
            frame = original()
            frame.received_monotonic_seconds -= 10
            return frame
        b.semantic_vision.fetch_frame = stale
    result = run(r)
    assert events[-1] == 'stop'
    if fault is None:
        assert len(motions(events)) == 1, json.dumps(result)
        obs = result['history'][0]['observation']
        assert obs['identity_source'] == 'marvin_locked_tracker_continuity'
        assert obs['semantic_gap_tracker_continuity']['semantic_refresh_required']
        assert obs['arrival']['association_forward_probe']['protected_radius_m'] == .45
        assert obs['source_frame_stamp_ns'] > seed['preview_result']['source_frame_stamp_ns']
        assert result['history'][0]['result']['motion_executed'] is True
    else:
        assert motions(events) == [] and result.get('actions_executed', 0) == 0
    assert r.mission_manager.get_active_mission() is None


def test_gap_observation_that_ages_before_dispatch_cannot_move(tmp_path, monkeypatch):
    r, b, _, events, clock = delayed_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .8), (0, .5)])
    b.observe_find_marvin_v2()
    def empty(*_args):
        raise ValueError('marvin_candidate_selection_response_empty')
    b.semantic_vision.select_marvin_candidate = empty
    dispatch = r._dispatch_marvin_observation_action
    def delayed_dispatch(*args, **kwargs):
        clock[0] += 1100000000
        return dispatch(*args, **kwargs)
    r._dispatch_marvin_observation_action = delayed_dispatch
    result = run(r)
    assert motions(events) == [] and result['actions_executed'] == 0
    assert events[-1] == 'stop'
