"""Default-off deployment contract. Offline fakes; no services or robot access."""
import ast
import copy
from dataclasses import fields, FrozenInstanceError
import importlib
import json
from pathlib import Path
import socket
import time

import pytest

MODULES = tuple('marvin_navigation_'+x for x in (
    'phases', 'policy', 'certificates', 'issuers', 'shadow',
    'instrumentation', 'shadow_runtime'))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline only: no network or service processes')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.delenv('MARVIN_NAVIGATION_SHADOW_ENABLED', raising=False)


@pytest.mark.parametrize('flag', [None, '', 'false', 'FALSE', '0', '1', 'yes', 'invalid'])
def test_default_off_never_constructs_payload_or_infrastructure(tmp_path, monkeypatch, flag):
    from test_marvin_local_bypass import mission_bundle
    import marvin_navigation_shadow_runtime as sr
    if flag is not None:
        monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_ENABLED', flag)
    monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_DIRECTORY', '/dev/null/navigation-shadow')
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled subsystem must do no work')
    monkeypatch.setattr(sr, 'from_environment', forbidden)
    (runtime, *_), _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=2)
    monkeypatch.setattr(runtime, '_emit_marvin_navigation_shadow', forbidden)
    from test_find_marvin_closed_loop import run
    result = run(runtime)
    assert result['state'] == 'ARRIVED'
    assert runtime._marvin_navigation_shadow is None
    assert runtime._marvin_navigation_shadow_service is None
    assert runtime._marvin_navigation_shadow_health() == {'enabled': False}
    assert runtime.get_status()['navigation_shadow'] == {'enabled': False}
    assert runtime.get_status_summary()['navigation_shadow'] == {'enabled': False}


@pytest.mark.parametrize('name', MODULES)
def test_runtime_dependency_isolation(name):
    tree = ast.parse(Path(name+'.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            assert not module.startswith('test_')
            assert not any(x in module for x in ('replay', 'calibration', 'validation'))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert '/tmp/mayday-' not in node.value
    importlib.import_module(name)


@pytest.mark.parametrize('name', MODULES)
def test_no_shadow_motion_endpoint_or_executor_reference(name):
    tree = ast.parse(Path(name+'.py').read_text())
    forbidden = {'move_forward', 'move_lateral', 'move_backward', 'turn_left', 'turn_right',
        'submit_intent', 'submit_mission', 'publish', 'dispatch', 'stop',
        'execute_single_marvin_approach_step', 'execute_single_marvin_alignment_step',
        'execute_single_marvin_lateral_step', '_execute_single_marvin_approach_step',
        '_consume_marvin_source_stamp'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert '/cmd_vel' not in node.value
            assert '/missions' not in node.value


def test_certificate_schema_cannot_contain_transport_handles():
    from marvin_navigation_certificates import (ObservationCertificate, GeometryCertificate,
        CompletionCertificate, JitVetoCertificate)
    forbidden = {'motion_authority', 'executor', 'dispatch_callback', 'bridge_client',
        'transport_handle', 'jit_grant', 'command_token'}
    for cls in (ObservationCertificate, GeometryCertificate, CompletionCertificate, JitVetoCertificate):
        assert not forbidden.intersection(f.name for f in fields(cls))


def test_every_common_intent_is_frozen_and_has_no_authority():
    from marvin_navigation_policy import IntentKind, NavigationIntent
    for kind in IntentKind:
        intent = NavigationIntent(kind, 'offline')
        assert intent.motion_authority is False
        with pytest.raises((FrozenInstanceError, AttributeError)):
            intent.motion_authority = True


def test_provisional_resource_limits_preserved():
    from marvin_navigation_shadow_runtime import Resources
    r = Resources()
    assert (r.capacity, r.critical_reserve, r.file_count, r.file_bytes) == (32, 8, 3, 1048576)
    assert r.max_record_bytes == 65536
    assert r.max_nodes == 6144 and r.max_cloud_points == 512
    assert r.shutdown_join_seconds == .1 and r.shutdown_drain_seconds == .05


def test_only_worker_evaluates_shadow_and_serializes():
    tree = ast.parse(Path('runtime.py').read_text())
    hooks = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name in ('_emit_marvin_navigation_shadow', '_capture_marvin_shadow_lidar_reference')]
    for hook in hooks:
        calls = [ast.unparse(n.func) for n in ast.walk(hook) if isinstance(n, ast.Call)]
        assert not any(any(x in call for x in ('dump', 'write', 'phase_policy', 'process', 'join')) for call in calls)


def test_disabled_status_does_not_instantiate_shadow():
    from runtime import CognitiveRuntime
    r = object.__new__(CognitiveRuntime)
    assert r._marvin_navigation_shadow_health() == {'enabled': False}


def test_worker_failure_does_not_mutate_native_return_or_consumed_stamp(tmp_path, monkeypatch):
    import marvin_navigation_shadow_runtime as sr
    from test_marvin_local_bypass import mission_bundle
    from test_find_marvin_closed_loop import run, motions
    monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_ENABLED', 'true')
    services = []
    def factory():
        s = sr.ShadowRuntime(str(tmp_path/'shadow'))
        def fail():
            raise OSError('offline simulated disk failure')
        s._new_sink = fail
        services.append(s)
        s.start()
        return s
    monkeypatch.setattr(sr, 'from_environment', factory)
    (r, _, _, events, _), _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=2)
    result = run(r)
    assert result['state'] == 'ARRIVED'
    assert result['local_avoidance_actions'] == 3 and result['local_bypass_actions'] == 2
    assert motions(events)[:3] == [('strafe', .08, 1.), ('forward', .1, .5), ('forward', .1, .5)]
    assert len(r._marvin_alignment_consumed_source_frame_stamps) == len(motions(events))
    for service in services:
        service.shutdown()
        assert not service.healthy and not service._thread.is_alive()


def test_capture_flag_initialization_failure_does_not_abort_runtime_startup(tmp_path, monkeypatch):
    import marvin_navigation_shadow_runtime as sr
    from test_find_marvin_closed_loop import Perception, run
    from test_marvin_local_bypass import mission_bundle
    original = Perception.__setattr__
    def fail_diagnostic_flag(self, name, value):
        if name == '_marvin_shadow_capture_enabled':
            raise ValueError('offline simulated diagnostic initialization failure')
        return original(self, name, value)
    service = sr.ShadowRuntime(str(tmp_path/'shadow'))
    monkeypatch.setattr(Perception, '__setattr__', fail_diagnostic_flag)
    monkeypatch.setattr(sr, 'from_environment', lambda: service)
    monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_ENABLED', 'true')
    (r, *_), _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=2)
    assert r._marvin_navigation_shadow is None
    assert r._marvin_navigation_shadow_service is None
    assert r._marvin_navigation_shadow_startup_error == 'ValueError'
    assert not service.buffer.statistics()['accepting']
    assert run(r)['state'] == 'ARRIVED'
    assert r._marvin_navigation_shadow_health() == {'enabled': False}
    assert not (tmp_path/'shadow').exists()
