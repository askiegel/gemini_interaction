"""Offline tests: all camera and model transports are replaced."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from behavior_manager import BehaviorManager
from semantic_vision import JpegFrame, MARVIN_DESCRIPTION, SemanticVisionClient

# Minimal SOF header fixture: parser tests do not perform image decoding.
JPEG = b'\xff\xd8\xff\xc0\x00\x0b\x08\x01\xe0\x02\x80\x01\x01\x11\x00\xff\xd9'
FRAME = JpegFrame(JPEG, 640, 480, '2026-09-15T12:00:00+00:00')


def payload(**updates):
    return dict(dict(target='backpack', found=True, coarse_direction='RIGHT',
                     image_width=640, image_height=480), **updates)


def helper(value=None):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=value, text=json.dumps(value))
    return SemanticVisionClient(client=client, model='configured-model', camera_url='http://camera.invalid/test.jpg'), client


def test_one_jpeg_prompt_schema_and_sdk_request_bounds():
    value, client = helper(payload(bbox=dict(x1=-2, y1=10, x2=700, y2=500)))
    result = value.describe(' Backpack ', FRAME)
    call = client.models.generate_content.call_args.kwargs
    assert client.models.generate_content.call_count == 1
    assert call['model'] == 'configured-model'
    assert len(call['contents']) == 2
    assert '"backpack"' in call['contents'][0]
    assert 'found=false' in call['contents'][0]
    assert call['contents'][1].inline_data.data == JPEG
    assert call['contents'][1].inline_data.mime_type == 'image/jpeg'
    config = call['config']
    assert config.response_mime_type == 'application/json'
    assert config.response_schema['required']
    assert config.http_options.timeout == 12000
    assert value.timeout_seconds == 5.0
    assert BehaviorManager.SEMANTIC_FRAME_TIMEOUT_SECONDS == 5.0
    assert BehaviorManager.SEMANTIC_IMAGE_TIMEOUT_SECONDS == 13.0
    assert (
        BehaviorManager.SEMANTIC_IMAGE_TIMEOUT_SECONDS
        > config.http_options.timeout / 1000
    )
    assert config.http_options.retry_options.attempts == 1
    assert config.automatic_function_calling.disable is True
    assert result['bbox'] == dict(x1=0, y1=10, x2=640, y2=480)
    assert result['frame_received_at'] == FRAME.received_at
    assert result['source'] == 'gemini_semantic'
    assert result['geometry_quality'] == 'coarse'
    assert not {'confidence', 'track_id', 'entity_id', 'identity_id', 'data'} & result.keys()


def test_marvin_uses_physical_semantic_specification_and_geometry():
    value, client = helper(dict(
        target='marvin', found=True, coarse_direction='LEFT',
        bbox=dict(x1=10, y1=20, x2=110, y2=220),
        image_width=640, image_height=480,
    ))
    result = value.describe('marvin', FRAME)
    prompt = client.models.generate_content.call_args.kwargs['contents'][0]
    assert MARVIN_DESCRIPTION in prompt
    assert "not the text label 'teddy bear'" in prompt
    assert 'Ignore humans in the frame.' in prompt
    assert result['source'] == 'gemini_marvin'
    assert result['semantic_target'] == 'marvin'
    assert result['source_timestamp'] == FRAME.received_at
    assert result['semantic_completed_at']
    assert result['semantic_completed_at'] != result['source_timestamp']
    assert result['center_x'] == 60.0
    assert result['center_y'] == 120.0


@pytest.mark.parametrize('value', [
    dict(target='marvin', found=True, coarse_direction='CENTER', bbox=None,
         image_width=640, image_height=480),
    dict(target='marvin', found=True, coarse_direction='CENTER',
         bbox=dict(x1=20, y1=1, x2=2, y2=3), image_width=640, image_height=480),
    dict(target='marvin', found=True, coarse_direction='CENTER',
         bbox=dict(x1=-1, y1=1, x2=20, y2=30), image_width=640, image_height=480),
])
def test_marvin_invalid_or_missing_bbox_fails_closed(value):
    instance, _ = helper(value)
    with pytest.raises(ValueError):
        instance.describe('marvin', FRAME)


def test_marvin_absence_requires_and_preserves_a_diagnostic_reason():
    instance, _ = helper(dict(
        target='marvin', found=False, coarse_direction='UNKNOWN', bbox=None,
        image_width=640, image_height=480, reason='marvin is absent',
    ))
    assert instance.describe('marvin', FRAME)['diagnostic_reason'] == 'marvin is absent'
    instance, _ = helper(dict(
        target='marvin', found=False, coarse_direction='UNKNOWN', bbox=None,
        image_width=640, image_height=480,
    ))
    with pytest.raises(ValueError, match='semantic_absent_reason_required'):
        instance.describe('marvin', FRAME)


def test_marvin_source_freshness_is_checked_before_inference_and_result_age_afterward():
    manager = BehaviorManager(robot_client=object())
    source = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    frame = SimpleNamespace(received_at=source.isoformat())
    submitted = source + timedelta(seconds=2.9)
    completed = source + timedelta(seconds=6.5)
    result = {
        'source_timestamp': source.isoformat(),
        'semantic_completed_at': completed.isoformat(),
    }
    assert manager._marvin_source_frame_is_fresh(frame, now=submitted)
    assert manager._marvin_semantic_is_current(
        result, now=completed + timedelta(seconds=.1),
    )


def test_marvin_stale_missing_malformed_and_future_timestamps_fail_closed():
    manager = BehaviorManager(robot_client=object())
    now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    assert not manager._marvin_source_frame_is_fresh(
        SimpleNamespace(received_at=(now - timedelta(seconds=3.1)).isoformat()), now=now,
    )
    for timestamp in (None, 'not-a-timestamp', (now + timedelta(seconds=1)).isoformat()):
        assert not manager._marvin_source_frame_is_fresh(
            SimpleNamespace(received_at=timestamp), now=now,
        )
    assert not manager._marvin_semantic_is_current({
        'source_timestamp': now.isoformat(),
        'semantic_completed_at': (now - timedelta(seconds=3.1)).isoformat(),
    }, now=now)


@pytest.mark.parametrize('value', [
    None, [], {}, payload(found='true'), payload(target='chair'),
    payload(coarse_direction='right'), payload(image_width=None),
    payload(image_width=641), payload(image_height=True),
    payload(bbox={'x1': 0}), payload(bbox=dict(x1=20, y1=1, x2=2, y2=3)),
    payload(bbox=dict(x1=float('nan'), y1=1, x2=20, y2=3)),
    payload(bbox=dict(x1=0, y1=1, x2=float('inf'), y2=3)),
    payload(bbox=dict(x1=False, y1=1, x2=20, y2=3)),
    payload(found=False), payload(confidence=.9), payload(track_id=1),
    payload(entity_id='x'), payload(identity_id='x'),
])
def test_malformed_schema_rejected(value):
    instance, _ = helper(value)
    with pytest.raises((ValueError, TypeError)):
        instance.describe('backpack', FRAME)


def test_missing_dimensions_never_inferred():
    value = payload()
    del value['image_width']
    instance, _ = helper(value)
    with pytest.raises(ValueError):
        instance.describe('backpack', FRAME)


def test_json_text_response_and_absent_result():
    instance, client = helper()
    client.models.generate_content.return_value.text = json.dumps(payload(found=False, coarse_direction='UNKNOWN'))
    assert instance.describe('backpack', FRAME)['found'] is False


def test_invalid_json_and_model_timeout_have_no_retry():
    instance, client = helper()
    client.models.generate_content.return_value.text = '{bad'
    with pytest.raises(ValueError):
        instance.describe('backpack', FRAME)
    client.models.generate_content.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        instance.describe('backpack', FRAME)
    assert client.models.generate_content.call_count == 2


class Response:
    def __init__(self, content=JPEG, status=200, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {'Content-Type': 'image/jpeg'}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('offline HTTP error')

    def iter_content(self, chunk_size):
        yield self.content


def test_configured_camera_url_and_local_timestamp(monkeypatch):
    monkeypatch.setenv('VISION_CAMERA_URL', 'http://camera.invalid/configured.jpg')
    response = Response()
    get = Mock(return_value=response)
    monkeypatch.setattr('semantic_vision.requests.get', get)
    instance = SemanticVisionClient(client=None, model='test')
    frame = instance.fetch_frame()
    assert (frame.data, frame.width, frame.height) == (JPEG, 640, 480)
    assert frame.received_at.endswith('+00:00')
    get.assert_called_once_with('http://camera.invalid/configured.jpg', timeout=5.0, stream=True, allow_redirects=False)
    assert response.closed


@pytest.mark.parametrize('response', [Response(b''), Response(status=500),
    Response(status=302), Response(b'not jpeg'), Response(JPEG[:-2]),
    Response(headers={'Content-Type': 'image/png'}),
    Response(headers={'Content-Length': '99999999'}), Response(b'x'*100)])
def test_frame_errors_are_closed_and_not_retried(monkeypatch, response):
    get = Mock(return_value=response)
    monkeypatch.setattr('semantic_vision.requests.get', get)
    instance, _ = helper()
    instance.max_image_bytes = 80
    with pytest.raises((ValueError, requests.HTTPError)):
        instance.fetch_frame()
    assert get.call_count == 1
    assert response.closed


def test_frame_http_timeout_no_retries(monkeypatch):
    get = Mock(side_effect=requests.Timeout('offline timeout'))
    monkeypatch.setattr('semantic_vision.requests.get', get)
    instance, _ = helper()
    with pytest.raises(requests.Timeout):
        instance.fetch_frame()
    assert get.call_count == 1


def test_runtime_factory_uses_configured_key_model_without_command_provider(monkeypatch):
    from google import genai
    constructor = Mock()
    monkeypatch.setattr(genai, 'Client', constructor)
    instance = SemanticVisionClient.from_config({'api_key': 'test-only-key', 'model': 'configured-model'})
    constructor.assert_called_once_with(api_key='test-only-key')
    assert instance.model == 'configured-model'
    assert instance.client is constructor.return_value


def test_missing_camera_configuration_fails_without_http(monkeypatch):
    monkeypatch.delenv('VISION_CAMERA_URL', raising=False)
    get = Mock()
    monkeypatch.setattr('semantic_vision.requests.get', get)
    instance = SemanticVisionClient(client=None, model='test')
    with pytest.raises(ValueError, match='camera_url_not_configured'):
        instance.fetch_frame()
    get.assert_not_called()


@pytest.mark.parametrize('elapsed,expired', [(4.999, False), (5.0, True)])
def test_slow_frame_stream_expires_and_closes(monkeypatch, elapsed, expired):
    response = Response()
    get = Mock(return_value=response)
    monkeypatch.setattr('semantic_vision.requests.get', get)
    ticks = iter([0.0, elapsed, elapsed])
    monkeypatch.setattr('semantic_vision.time.monotonic', lambda: next(ticks))
    instance, _ = helper()
    assert instance.timeout_seconds == 5.0
    if expired:
        with pytest.raises(TimeoutError):
            instance.fetch_frame()
    else:
        assert instance.fetch_frame().data == JPEG
    assert get.call_count == 1 and response.closed


def test_runtime_explicitly_injects_independent_visual_helper(monkeypatch, tmp_path):
    import runtime
    from world_model import WorldModel
    config = {'api_key': 'test-only', 'model': 'configured-model'}
    monkeypatch.setattr(runtime, 'load_config', lambda: config)
    monkeypatch.setattr(runtime, 'create_provider', lambda _: object())
    visual = object()
    factory = Mock(return_value=visual)
    monkeypatch.setattr(runtime.SemanticVisionClient, 'from_config', factory)
    robot = SimpleNamespace(base_url='http://robot.invalid', stop=Mock(),
                            configure_forward_interlock=Mock())
    instance = runtime.CognitiveRuntime(
        world_model=WorldModel(str(tmp_path / 'world.json')),
        vision_adapter=object(), robot_client=robot,
        lidar_worker_factory=lambda *a, **kw: SimpleNamespace(session='offline'),
    )
    assert instance.behavior_manager.semantic_vision is visual
    factory.assert_called_once_with(config)
    robot.stop.assert_not_called()
