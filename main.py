"""
main.py — example entry point.

Demonstrates:
  1. Static registration of 2 cameras before start()
  2. Dynamic addition of a 3rd camera while running
  3. Graceful SIGINT / SIGTERM shutdown
  4. Printing results from all cameras in one loop

Run:
    python main.py

Or load configs from a JSON file:
    python main.py --config cameras.json

cameras.json format:
    {
      "cameras": [
        {
          "camera_id":    "cam_entry",
          "source":       "rtsp://192.168.1.10/stream1",
          "core_id":      0,
          "zone_file":    "/path/to/zones_runtime.json",
          "yolo_model":   "yolov8n_openvino_model",
          "gender_model": "/path/to/mobilenetv3_gender_best.pth",
          "show_window":  true
        },
        {
          "camera_id":    "cam_exit",
          "source":       "rtsp://192.168.1.11/stream1",
          "core_id":      1,
          ...
        }
      ]
    }
"""

import argparse
import signal
import sys
import time

from config import CameraConfig
from manager import PipelineManager

# ─── Shared paths (edit for your deployment) ──────────────────────────────────
_ZONE_FILE    = "/home/keshav/rajan/new_pipeline/zones_runtime.json"
_YOLO_MODEL   = "yolov8n_openvino_model"
_GENDER_MODEL = "/home/keshav/rajan/new_pipeline/models/mobilenetv3_gender_best.pth"


def build_default_configs() -> list[CameraConfig]:
    """Return two hardcoded camera configs for quick testing."""
    return [
        CameraConfig(
            camera_id    = "cam_entry",
            source       = "rtsp://192.168.1.10/stream1",
            core_id      = 0,
            zone_file    = _ZONE_FILE,
            yolo_model   = _YOLO_MODEL,
            gender_model = _GENDER_MODEL,
            show_window  = True,
        ),
        CameraConfig(
            camera_id    = "cam_exit",
            source       = "rtsp://192.168.1.11/stream1",
            core_id      = 1,
            zone_file    = _ZONE_FILE,
            yolo_model   = _YOLO_MODEL,
            gender_model = _GENDER_MODEL,
            show_window  = True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera pipeline manager")
    parser.add_argument("--config", default=None,
                        help="Path to cameras.json (optional)")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    if args.config:
        configs = CameraConfig.load_all(args.config)
        print(f"[main] loaded {len(configs)} camera(s) from {args.config}")
    else:
        configs = build_default_configs()
        print(f"[main] using {len(configs)} hardcoded camera config(s)")

    # ── Create manager and register cameras ───────────────────────────────────
    mgr = PipelineManager()
    for cfg in configs:
        mgr.add_camera(cfg)

    # ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────────────
    def _shutdown(sig, frame):
        print("\n[main] shutdown requested — stopping all cameras...")
        mgr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Launch all registered cameras ─────────────────────────────────────────
    mgr.start()

    # ── Example: add a 3rd camera dynamically after 15 seconds ───────────────
    # In production you'd trigger this from an API endpoint or config reload.
    _demo_add_at = time.time() + 15.0
    _demo_added  = False

    # ── Main results loop ─────────────────────────────────────────────────────
    print("[main] consuming results — press Ctrl+C to stop\n")

    for result in mgr.results():
        cam   = result["camera_id"]
        core  = result["core_id"]
        count = result["track_count"]
        ts    = result["timestamp"]

        # One-line summary per result
        zone_summary = ", ".join(
            f"{t['id']}@{','.join(t['zones']) or 'open'}"
            for t in result["tracks"]
        )
        print(f"[{cam}|core{core}] {count} people  {zone_summary}  ts={ts:.2f}")

        # ── Dynamic camera add demo ───────────────────────────────────────────
        if not _demo_added and time.time() >= _demo_add_at:
            next_core = mgr.auto_assign_core()
            new_cfg = CameraConfig(
                camera_id    = "cam_aisle",
                source       = 2,           # device index — change to an RTSP URL
                core_id      = next_core,
                zone_file    = _ZONE_FILE,
                yolo_model   = _YOLO_MODEL,
                gender_model = _GENDER_MODEL,
                show_window  = True,
            )
            mgr.add_camera(new_cfg)
            print(f"\n[main] dynamically added 'cam_aisle' on core {next_core}\n")
            _demo_added = True

        # ── Periodic status print (every ~60 s via simple counter trick) ─────
        if int(ts) % 60 == 0:
            print("\n[main] status snapshot:", mgr.status(), "\n")


if __name__ == "__main__":
    main()
