"""
manager.py — PipelineManager

Orchestrates N camera worker processes, one per CPU core.

Quick start
───────────
    from config import CameraConfig
    from manager import PipelineManager

    mgr = PipelineManager()

    # Static registration before start()
    mgr.add_camera(CameraConfig(
        camera_id="cam_entry", source="rtsp://10.0.0.1/stream",
        core_id=0, zone_file="zones.json",
        yolo_model="yolov8n_openvino_model",
        gender_model="models/gender.pth",
    ))
    mgr.add_camera(CameraConfig(
        camera_id="cam_exit", source="rtsp://10.0.0.2/stream",
        core_id=1, ...
    ))

    mgr.start()

    # Dynamic addition while running:
    mgr.add_camera(CameraConfig(
        camera_id="cam_aisle", source=2,
        core_id=mgr.auto_assign_core(), ...
    ))

    for result in mgr.results():     # blocking generator
        print(result["camera_id"], result["track_count"])

    mgr.stop()

Design notes
────────────
- multiprocessing.get_context("spawn") is mandatory: 'fork' copies the parent's
  OpenVINO / CUDA state into each child and causes segfaults or silent wrong
  results.  'spawn' starts a clean interpreter every time.

- Each worker process is a daemon (daemon=True) so the OS reaps them if the
  parent dies unexpectedly.

- The watchdog thread polls every 2 s and restarts crashed workers up to
  _MAX_RESTARTS times.  After that it logs an error and leaves the slot empty
  so the rest of the pipeline keeps running.

- result_queue is shared across all workers.  Each put is non-blocking; slow
  consumers cause drops in worker.py, never process stalls.

- _lock only protects the _workers dict and _used_cores set — it is never held
  during blocking I/O or process joins to avoid deadlocks.
"""

import multiprocessing as mp
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from config import CameraConfig
from worker import _worker_main

_RESTART_DELAY = 3.0   # seconds to wait before restarting a crashed worker
_MAX_RESTARTS  = 5     # give up after this many consecutive crashes


@dataclass
class _WorkerState:
    config:        CameraConfig
    process:       Optional[mp.Process]
    stop_event:    mp.Event              # type: ignore[type-arg]
    restart_count: int   = 0
    started_at:    float = field(default_factory=time.time)


class PipelineManager:
    """
    Manages N camera processes dynamically.

    Public API
    ──────────
    auto_assign_core()         → next unused core id (int)
    add_camera(cfg)            → register + optionally launch a camera
    remove_camera(camera_id)   → gracefully stop and deregister
    start()                    → launch all registered cameras + watchdog
    stop()                     → signal all workers, join, cleanup
    status()                   → dict snapshot of all worker states
    results()                  → blocking generator of result dicts
    """

    def __init__(self) -> None:
        # 'spawn' starts a fresh Python interpreter in each child.
        # This is mandatory for torch/OpenVINO correctness and safety.
        self._ctx = mp.get_context("spawn")

        self._workers:    dict[str, _WorkerState] = {}
        self._used_cores: set[int]                = set()

        # All workers share one result queue.  maxsize caps memory usage.
        self._result_q: mp.Queue = self._ctx.Queue(maxsize=1000)

        self._lock    = threading.Lock()
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def auto_assign_core(self) -> int:
        """
        Return the lowest core id that is not already in use.
        Thread-safe: call from any thread at any time.
        """
        cid = 0
        with self._lock:
            while cid in self._used_cores:
                cid += 1
        return cid

    def add_camera(self, cfg: CameraConfig) -> None:
        """
        Register a camera and, if the manager is already running, launch it
        immediately on its assigned core.

        If the manager has not been started yet the camera is queued and
        launched when start() is called.

        Raises ValueError if camera_id is already registered.
        """
        with self._lock:
            if cfg.camera_id in self._workers:
                raise ValueError(
                    f"Camera '{cfg.camera_id}' is already registered. "
                    f"Call remove_camera() first."
                )
            self._used_cores.add(cfg.core_id)
            # Store a placeholder — process=None means not yet launched
            self._workers[cfg.camera_id] = _WorkerState(
                config=cfg,
                process=None,
                stop_event=self._ctx.Event(),
            )

        if self._running:
            self._launch_worker(cfg)

    def remove_camera(self, camera_id: str, timeout: float = 5.0) -> None:
        """
        Stop and deregister a camera.  Safe to call while the manager is
        running.  Blocks up to `timeout` seconds waiting for the process to
        exit, then force-kills it.
        """
        with self._lock:
            state = self._workers.pop(camera_id, None)
            if state is None:
                return
            self._used_cores.discard(state.config.core_id)

        state.stop_event.set()
        if state.process is not None and state.process.is_alive():
            state.process.join(timeout=timeout)
            if state.process.is_alive():
                state.process.terminate()
                state.process.join(timeout=2.0)

        print(f"[manager] removed '{camera_id}'", flush=True)

    def start(self) -> None:
        """
        Launch all registered cameras and start the watchdog thread.
        Idempotent — calling start() twice does nothing the second time.
        """
        if self._running:
            return
        self._running = True

        with self._lock:
            pending_configs = [s.config for s in self._workers.values()]

        for cfg in pending_configs:
            self._launch_worker(cfg)

        self._watchdog_thread = threading.Thread(
            target=self._watchdog, daemon=True, name="mgr-watchdog"
        )
        self._watchdog_thread.start()
        print(f"[manager] started — {len(pending_configs)} camera(s)", flush=True)

    def stop(self, timeout: float = 10.0) -> None:
        """
        Gracefully stop all workers, then drain the result queue.
        Blocks until all processes have exited or the timeout expires.
        """
        self._running = False

        with self._lock:
            states = list(self._workers.values())

        for state in states:
            state.stop_event.set()

        deadline = time.time() + timeout
        for state in states:
            if state.process is None:
                continue
            remaining = max(0.0, deadline - time.time())
            state.process.join(timeout=remaining)
            if state.process.is_alive():
                print(f"[manager] force-killing '{state.config.camera_id}'",
                      flush=True)
                state.process.terminate()
                state.process.join(timeout=2.0)

        # Drain so the queue's background thread can exit cleanly
        try:
            while not self._result_q.empty():
                self._result_q.get_nowait()
        except Exception:
            pass

        print("[manager] all workers stopped", flush=True)

    def status(self) -> dict[str, dict]:
        """
        Return a snapshot of all worker states.  Thread-safe read.

        Example return value:
        {
          "cam_entry": {"alive": True,  "core_id": 0, "restarts": 0, "source": "rtsp://..."},
          "cam_exit":  {"alive": False, "core_id": 1, "restarts": 2, "source": 1},
        }
        """
        with self._lock:
            return {
                cid: {
                    "alive":    state.process.is_alive() if state.process else False,
                    "core_id":  state.config.core_id,
                    "restarts": state.restart_count,
                    "source":   state.config.source,
                    "pid":      state.process.pid if state.process else None,
                }
                for cid, state in self._workers.items()
            }

    def results(self) -> Iterator[dict]:
        """
        Blocking generator that yields result dicts from all running cameras.

        Each dict has the shape:
        {
          "camera_id":   str,
          "core_id":     int,
          "timestamp":   float,
          "track_count": int,
          "tracks": [{"id": int, "gender": str, "zones": list[str]}, ...],
        }

        The generator exits when stop() is called (self._running becomes False
        and no result arrives within 1 second).
        """
        while self._running:
            try:
                yield self._result_q.get(timeout=1.0)
            except Exception:
                continue   # timeout — loop back and check _running

        # Drain any remaining results after stop()
        try:
            while True:
                yield self._result_q.get_nowait()
        except Exception:
            return

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _launch_worker(self, cfg: CameraConfig) -> None:
        """
        Spawn a new process for `cfg`.  If a previous _WorkerState exists for
        this camera_id its stop_event and restart_count are preserved.
        """
        with self._lock:
            old = self._workers.get(cfg.camera_id)
            stop_event    = old.stop_event    if old else self._ctx.Event()
            restart_count = old.restart_count if old else 0

        stop_event.clear()   # reset in case of restart

        proc = self._ctx.Process(
            target  = _worker_main,
            args    = (cfg.to_dict(), self._result_q, stop_event),
            name    = f"cam-{cfg.camera_id}",
            daemon  = True,
        )
        proc.start()

        with self._lock:
            self._workers[cfg.camera_id] = _WorkerState(
                config        = cfg,
                process       = proc,
                stop_event    = stop_event,
                restart_count = restart_count,
                started_at    = time.time(),
            )

        print(
            f"[manager] launched '{cfg.camera_id}' "
            f"pid={proc.pid} core={cfg.core_id}",
            flush=True,
        )

    def _watchdog(self) -> None:
        """
        Polls every 2 seconds.  For any worker that has died unexpectedly
        (i.e. stop_event was NOT set), attempts a restart up to _MAX_RESTARTS.

        Runs as a daemon thread — exits automatically when the main process
        exits.  Uses only the public _workers / stop_event fields, never
        the process handle directly except to check is_alive() and exitcode.
        """
        while self._running:
            time.sleep(2.0)

            with self._lock:
                camera_ids = list(self._workers.keys())

            for cid in camera_ids:
                with self._lock:
                    state = self._workers.get(cid)
                if state is None:
                    continue
                if state.process is None:
                    continue
                if state.stop_event.is_set():
                    continue   # intentional stop — don't restart

                if not state.process.is_alive():
                    exit_code = state.process.exitcode

                    if state.restart_count >= _MAX_RESTARTS:
                        print(
                            f"[manager][watchdog] '{cid}' is permanently down "
                            f"(crashed {state.restart_count}x, exit={exit_code}). "
                            f"Call add_camera() with a fresh config to re-add it.",
                            file=sys.stderr, flush=True,
                        )
                        continue

                    print(
                        f"[manager][watchdog] '{cid}' crashed (exit={exit_code}) — "
                        f"restart {state.restart_count + 1}/{_MAX_RESTARTS} "
                        f"in {_RESTART_DELAY}s",
                        flush=True,
                    )
                    time.sleep(_RESTART_DELAY)

                    # Increment restart counter before re-launching
                    with self._lock:
                        if cid in self._workers:
                            self._workers[cid].restart_count += 1
                            cfg = self._workers[cid].config
                        else:
                            continue   # was removed while we waited

                    self._launch_worker(cfg)
