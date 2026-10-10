"""Bounded, passive LIVE SHADOW design candidate; default OFF.

Navigation intent is never motion authority. Only the daemon diagnostic worker
constructs certificates, evaluates phases, serializes or performs disk I/O.
The producer owns its input dictionaries until bounded structural capture ends.
No runtime, Bridge, executor, sensor getter, authority ledger or callback is
stored in this service. CPython/GIL counter semantics match PassiveBuffer.
Resource values below are PROVISIONAL, requiring robot timing acceptance.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time

from marvin_navigation_instrumentation import (
    PassiveBuffer, Limits, DiagnosticEvent, FrozenMap, EventType, thaw,
    serialize_event, freeze,
)

# Only irrelevant bulky data is omitted. Native bounded geometry.points remains
# transiently necessary to the EXISTING issuer's safety-envelope revalidation.
# It is never written to disk, never truncated and never granted authority.
OMIT_CAPTURE=frozenset({'raw_image','camera_image','image_base64','raw_ranges',
    'raw_scan','sensor_msgs','mission_history','full_runtime_dump','image',
    'raw_camera','camera_frame','encoded_image','ranges','scan_ranges','geometry'})
# The current issuer reads canonical local_motion_geometry from the raw scan.
# Duplicate safety-result .geometry dictionaries are not certificate inputs;
# their permitted/radius/sector verdicts remain. Retain just one native XY cloud.
OMIT_OUTPUT=OMIT_CAPTURE|{'points','ranges','image','raw_camera','lidar_snapshot'}


@dataclass(frozen=True,slots=True)
class Resources:
    capacity:int=32
    correlation_capacity:int=4
    critical_reserve:int=8
    max_nodes:int=6144
    max_text_bytes:int=32768
    max_depth:int=24
    max_cloud_points:int=512
    max_record_bytes:int=65536
    batch_size:int=8
    wake_seconds:float=.05
    file_bytes:int=1048576
    file_count:int=3
    shutdown_drain_seconds:float=.05
    shutdown_join_seconds:float=.10

    def __post_init__(self):
        for name in ('capacity','correlation_capacity','max_nodes','max_text_bytes','max_depth','max_cloud_points',
                     'max_record_bytes','batch_size','file_bytes','file_count'):
            x=getattr(self,name)
            if type(x) is not int or x<=0:raise ValueError('positive bounded '+name)
        if type(self.critical_reserve) is not int or not 0<=self.critical_reserve<self.capacity:raise ValueError('reserve')
        if self.batch_size>self.capacity or self.max_record_bytes>self.file_bytes:raise ValueError('bounded batch/record')
        import math
        for name in ('wake_seconds','shutdown_drain_seconds','shutdown_join_seconds'):
            x=getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=1:raise ValueError('bounded lifecycle '+name)
        if self.shutdown_drain_seconds>self.shutdown_join_seconds:raise ValueError('drain <= join')

    @property
    def limits(self):return Limits(self.capacity,self.max_nodes,self.max_text_bytes,self.max_depth)

    @property
    def capture_json_upper_bound_bytes(self):
        # Includes structural punctuation, <=128-bit ints/finite floats,
        # worst ASCII escaping (6x), and bounded event-envelope identifiers.
        # Not serialized on producer path; this is a conservative byte bound.
        return self.max_nodes*64+self.max_text_bytes*6+4096

    @classmethod
    def diagnostic_rich(cls):
        return cls(capacity=64,critical_reserve=16,max_nodes=8192,max_text_bytes=65536,
            max_depth=32,max_cloud_points=1024,max_record_bytes=131072,batch_size=16,
            file_bytes=2097152,file_count=4)


def flag_enabled(value):
    """Absent/malformed fail safely OFF; only explicit textual true enables."""
    return type(value) is str and value.strip().lower()=='true'


def evidence_identifiers(event):
    """Worker-only primitive provenance, including failed issuance boundaries.

    Missing stays None; actual zero/false is preserved. These identifiers are
    never authority and never substitute for a valid certificate.
    """
    p=thaw(event.payload)
    observation=p.get('observation') or {}
    result=p.get('result') or {}
    planning=p.get('lidar') or {}
    pair=p.get('geometry_pair')
    if not planning and pair:planning=pair[0] or {}
    jit=((result.get('pre_transport_jit_veto') or {}).get('lidar_snapshot') or {})
    pair=p.get('jit_pair')
    if not jit and pair:jit=pair[0] or {}
    stamp=p.get('source_frame_stamp_ns')
    if stamp is None:stamp=observation.get('source_frame_stamp_ns')
    if stamp is None:stamp=result.get('source_frame_stamp_ns')
    return dict(source_frame_stamp_ns=stamp,expected_producer_session=p.get('producer_session'),
        planning_producer_session=planning.get('producer_session'),
        planning_sequence=p.get('planning_sequence') if p.get('planning_sequence') is not None else planning.get('acquisition_sequence'),
        jit_producer_session=jit.get('producer_session'),
        jit_sequence=p.get('jit_sequence') if p.get('jit_sequence') is not None else jit.get('acquisition_sequence'),
        source_stamp_consumed=p.get('source_stamp_consumed'),
        camera_floor_ns=p.get('camera_floor_ns'),previous_source_frame_stamp_ns=p.get('previous_source_frame_stamp_ns'))


class RotatingJsonl:
    """Worker-only I/O. One nonblocking OS lease per output directory.

    current=000, previous=001..N-1. Retain only these owned names, never delete
    unrelated files. A line larger than the bound is rejected, not truncated.
    No fsync on the runtime thread (indeed no runtime thread file access).
    """
    def __init__(self,directory,resources):
        import fcntl
        self.resources=resources;self.directory=Path(directory)
        if not self.directory.is_absolute():raise ValueError('absolute output directory required')
        self.directory.mkdir(parents=True,exist_ok=True)
        self._lease=None;self._file=None;self.bytes=0
        lease=self.directory/'marvin-shadow.lock'
        if lease.is_symlink():raise ValueError('symlink lease')
        self._lease=open(lease,'ab',buffering=0)
        try:
            fcntl.flock(self._lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            for i in range(resources.file_count):
                p=self.path(i)
                if p.is_symlink() or p.exists() and not p.is_file():raise ValueError('regular owned files only')
                if p.exists() and p.stat().st_size>resources.file_bytes:raise ValueError('preexisting file exceeds selected profile')
            # A profile change must not quietly violate retention or delete
            # an existing richer recording. Require operator-managed migration.
            for i in range(resources.file_count,4):
                if self.path(i).exists():raise ValueError('preexisting richer profile requires explicit retention migration')
            self._open()
        except BaseException:
            self.close();raise

    def path(self,index=0):return self.directory/('marvin-shadow-%03d.jsonl'%index)

    def _open(self):
        self._file=open(self.path(),'ab+',buffering=0)
        self.bytes=self._file.tell()
        if self.bytes>self.resources.file_bytes:self._rotate()
        elif self.bytes:
            self._file.seek(-1,os.SEEK_END)
            tail=self._file.read(1)
            self._file.seek(0,os.SEEK_END)
            if tail!=b'\n':
                # Preserve a crashed/partial tail in the older diagnostic file;
                # never concatenate a fresh event onto an incomplete JSON row.
                self._rotate()

    def _rotate(self):
        self._file.close();self._file=None
        oldest=self.path(self.resources.file_count-1)
        if oldest.exists():oldest.unlink()
        for i in range(self.resources.file_count-2,-1,-1):
            p=self.path(i)
            if p.exists():p.replace(self.path(i+1))
        self._open()

    def write(self,line):
        if type(line) is not bytes or len(line)>self.resources.max_record_bytes:raise ValueError('record byte bound')
        if self.bytes+len(line)>self.resources.file_bytes:self._rotate()
        # Partial write becomes diagnostic failure; no retry or control effect.
        n=self._file.write(line)
        self.bytes+=n
        if n!=len(line):raise OSError('partial diagnostic write')

    def close(self):
        if self._file is not None:
            try:self._file.close()
            finally:self._file=None
        if self._lease is not None:
            try:self._lease.close()
            finally:self._lease=None


class ShadowRuntime:
    """Single daemon worker. Neither output nor failures can reach executors.

    A hung OS write cannot be forcibly cancelled safely by Python; bounded join
    abandons that daemon. No replacement writer/retry is spawned. A service
    process exit can terminate it; live deployment must measure worker behavior.
    """
    def __init__(self,output_directory,*,resources=None):
        self.resources=resources or Resources()
        self.buffer=PassiveBuffer(self.resources.limits,priority=True,
            reserve=self.resources.critical_reserve,omit_keys=OMIT_CAPTURE,
            max_points=self.resources.max_cloud_points,points_xy_only=True)
        self.output_directory=output_directory
        self._stop=threading.Event();self._thread=None;self._deadline=None
        self._start_attempted=False
        self.enabled=True;self.healthy=False;self.writer_errors=0;self.processing_errors=0
        self.events_written=0;self.last_event_index=None;self.current_output_file=None
        self.error=None;self.abandoned=False;self.record_index=0
        self.stream_gaps=0;self.correlation_evictions=0
        self.certificate_errors=0;self.correlation_eviction_records=[]
        self.run_id=None

    def start(self):
        if self._start_attempted:return False
        self._start_attempted=True
        try:
            self._thread=threading.Thread(target=self._run,name='marvin-navigation-shadow',daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self._thread=None
            self._disable(exc);return False

    def _disable(self,exc):
        self.error=type(exc).__name__;self.healthy=False;self.enabled=False
        self.buffer.close(reason='writer_unavailable')

    def _new_consumer(self):
        # Imports/phase construction occur ONLY here on diagnostic worker.
        from marvin_navigation_instrumentation import OfflineShadowConsumer
        from marvin_navigation_phases import Watchdogs
        r=self.resources
        return OfflineShadowConsumer(limits=Limits(r.correlation_capacity,r.max_nodes,r.max_text_bytes,r.max_depth),
            phase_config=Watchdogs(.16))

    def _new_sink(self):return RotatingJsonl(self.output_directory,self.resources)

    def _record(self,sink,event):
        # Disk contains compact derived certificates/comparisons, never clouds.
        compact=freeze(thaw(event.payload),self.resources.limits,omit_keys=OMIT_OUTPUT)
        clean=DiagnosticEvent(event.schema_version,event.mission_id,event.event_type,
            event.event_index,event.event_time,event.provenance,compact)
        self.record_index+=1
        import json
        row=json.loads(serialize_event(clean))
        row['writer_record_index']=self.record_index
        row['diagnostic_run_id']=self.run_id or 'OFFLINE_UNSTARTED_HARNESS'
        row['ordering_semantics']='per diagnostic run admission order; original timestamp; flush is not control order'
        row['motion_authority']=False
        line=(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode('utf8')
        if len(line)>self.resources.max_record_bytes:
            self.processing_errors+=1;return
        sink.write(line);self.events_written+=1
        self.last_event_index=event.event_index

    def _run(self):
        sink=None
        try:
            import uuid
            self.run_id=uuid.uuid4().hex  # Worker only; indexes restart per run.
            consumer=self._new_consumer();sink=self._new_sink()
            self.current_output_file=str(sink.path());self.healthy=True
            while True:
                if self._stop.is_set() and time.monotonic()>=self._deadline:break
                events=self.buffer.drain(self.resources.batch_size)
                if not events:
                    if self._stop.is_set():break
                    self._stop.wait(self.resources.wake_seconds);continue
                for event in events:
                    if self._stop.is_set() and time.monotonic()>=self._deadline:
                        # The already-drained batch is still bounded. Account
                        # for unprocessed records rather than hiding their loss.
                        remaining=events[events.index(event):]
                        for lost in remaining:
                            next(self.buffer._full);next(self.buffer._drop_reasons['shutdown_discard'])
                            self.buffer._recent_drops.append(dict(event_index=lost.event_index,event_type=lost.event_type.value,
                                reason='shutdown_discard',detail='drain deadline'))
                        break
                    try:
                        before=consumer.failures
                        consumer.process(event)
                        self.processing_errors+=consumer.failures-before
                        self.certificate_errors=consumer.failures
                        self.stream_gaps=consumer.stream_gaps
                        self.correlation_evictions=consumer.correlation_evictions
                        self.correlation_eviction_records=list(consumer.correlation_eviction_records)
                    except Exception as exc:
                        self.processing_errors+=1
                        # Never restart correlation/reducer state after an
                        # uncaught failure: disable this diagnostic session.
                        raise RuntimeError('shadow consumer failed') from exc
                    # Envelope gives a compact audit trail even when processing
                    # has no certificate. Never serialize original raw payload.
                    envelope=DiagnosticEvent(1,event.mission_id,event.event_type,event.event_index,
                        event.event_time,event.provenance,freeze({'capture_only':True,
                        'evidence_processed':True,'identifiers':evidence_identifiers(event),
                        'native_geometry_projection':'XY_ONLY; no point truncation or coordinate recomputation'},self.resources.limits))
                    self._record(sink,envelope)
                    while consumer.records:
                        record=consumer.records.popleft()
                        try:self._record(sink,record)
                        except (ValueError,TypeError,OverflowError):self.processing_errors+=1
                # A bounded batch yields even during sustained floods; fresh
                # producer events are never a reason to monopolize the CPU.
                if not self._stop.is_set():self._stop.wait(self.resources.wake_seconds)
        except Exception as exc:
            self.writer_errors+=1;self._disable(exc)
        finally:
            self.buffer.close();self.healthy=False
            if sink is not None:
                try:sink.close()
                except Exception as exc:self.writer_errors+=1;self._disable(exc)
            # Do not retain queued evidence after normal/failing termination.
            self.buffer.discard()

    def request_shutdown(self):
        """Stop accepting immediately; no join on the signal/STOP path."""
        self.buffer.close()
        if self._deadline is None:self._deadline=time.monotonic()+self.resources.shutdown_drain_seconds
        self._stop.set()

    def shutdown(self):
        """Caller never waits for disk/queue locks; join has an explicit bound."""
        self.request_shutdown()
        t=self._thread
        if t is not None and t is not threading.current_thread():
            t.join(self.resources.shutdown_join_seconds)
            self.abandoned=t.is_alive()
        self.enabled=False
        if self.abandoned:self.healthy=False;self.error='ABANDONED_BLOCKED_WRITER'

    def status(self):
        s=self.buffer.statistics()
        drops=s['full_drops']+s['contention_drops']+s['construction_failures']+s['evicted_events']
        degraded=bool(drops or self.processing_errors or self.stream_gaps or self.correlation_evictions)
        state='DISABLED' if not self.enabled and not self.error else 'FAILED' if self.error or self.writer_errors else 'DEGRADED' if degraded else 'HEALTHY' if self.healthy else 'STARTING'
        return dict(enabled=self.enabled,healthy=self.healthy and not degraded,
            runtime_writer_healthy=self.healthy,coverage_degraded=degraded,health_state=state,
            drops_present=bool(drops),certificate_errors_present=bool(self.certificate_errors),
            certificate_errors=self.certificate_errors,
            drop_reasons=s['drop_reasons'],recent_drops=s['recent_drops'],
            stream_gaps=self.stream_gaps,correlation_evictions=self.correlation_evictions,
            correlation_eviction_records=self.correlation_eviction_records,
            queue_depth=s['queued'],queue_capacity=s['capacity'],
            queue_high_water=s['high_water'],
            dropped_events=drops,
            writer_errors=self.writer_errors,processing_errors=self.processing_errors,
            events_written=self.events_written,last_event_index=self.last_event_index,
            current_output_file=self.current_output_file,abandoned=self.abandoned,
            diagnostic_run_id=self.run_id,
            error=self.error,motion_authority=False)


def from_environment(environ=None):
    """No file operations, queue or coordinator in disabled mode.

    Output path/profile are explicit future service configuration. The default
    conservative profile and bounded local retention are provisional (not archival).
    """
    env=os.environ if environ is None else environ
    if not flag_enabled(env.get('MARVIN_NAVIGATION_SHADOW_ENABLED')):return None
    profile=env.get('MARVIN_NAVIGATION_SHADOW_PROFILE','conservative')
    if profile not in ('conservative','diagnostic-rich'):return None
    resources=Resources() if profile=='conservative' else Resources.diagnostic_rich()
    service=ShadowRuntime(env.get('MARVIN_NAVIGATION_SHADOW_DIRECTORY','/home/tkieg/.local/state/mayday/navigation-shadow'),resources=resources)
    service.start()
    return service  # Disabled failure status remains visible without retries.
