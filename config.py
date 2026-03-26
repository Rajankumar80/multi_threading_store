"""
config.py — CameraConfig dataclass.

Every field that used to be a module-level constant in detector.py now lives
here so that PipelineManager can spin up N independent processes, each with
its own paths, core binding, and output files.

JSON format for load_all():
{
  "cameras": [
    {
      "camera_id":    "cam_entry",
      "source":       "rtsp://192.168.1.10/stream1",
      "core_id":      0,
      "zone_file":    "/path/to/zones_runtime.json",
      "yolo_model":   "yolov8n_openvino_model",
      "gender_model": "/path/to/mobilenetv3_gender_best.pth"
    },
    {
      "camera_id":    "cam_exit",
      "source":       1,
      "core_id":      1,
      ...
    }
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Union


@dataclass
class CameraConfig:
    # ── Required ──────────────────────────────────────────────────────────────
    camera_id:    str              # unique name used in logs, windows, output files
    source:       Union[str, int]  # rtsp:// URL, file path, or cv2 device index
    core_id:      int              # which logical CPU core to pin this process to
    zone_file:    str              # path to zones_runtime.json
    yolo_model:   str              # path/dir to OpenVINO model
    gender_model: str              # path to .pth MobileNetV3 gender weights

    # ── Hysteresis tuning ─────────────────────────────────────────────────────
    entry_frames: int = 4
    exit_frames:  int = 6

    # ── Gender inference tuning ───────────────────────────────────────────────
    gender_cache_ttl:   int   = 90    # frames before re-inference
    gender_conf_thresh: float = 0.75  # re-infer below this confidence
    gender_skip_frames: int   = 3     # only submit job every N frames

    # ── I/O flush intervals (seconds) ─────────────────────────────────────────
    events_flush_interval:  float = 5.0
    metrics_flush_interval: float = 60.0

    # ── Display ───────────────────────────────────────────────────────────────
    show_window: bool = True   # False for headless / server deployments

    # ── Output files (auto-named from camera_id if not specified) ─────────────
    metrics_file: Optional[str] = None
    events_file:  Optional[str] = None

    def __post_init__(self) -> None:
        if self.metrics_file is None:
            self.metrics_file = f"{self.camera_id}_metrics.json"
        if self.events_file is None:
            self.events_file = f"{self.camera_id}_events.json"
        # Coerce string "0", "1" etc. to int for device-index sources
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

    # ── Serialisation helpers (needed to pass config across process boundary) ─

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CameraConfig:
        # Filter to only known fields so forward-compatible JSON doesn't crash
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load_all(cls, path: str) -> list[CameraConfig]:
        """Load a list of CameraConfigs from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [cls.from_dict(c) for c in data["cameras"]]

    @staticmethod
    def save_all(configs: list[CameraConfig], path: str) -> None:
        """Persist a list of CameraConfigs back to JSON."""
        with open(path, "w") as f:
            json.dump({"cameras": [c.to_dict() for c in configs]}, f, indent=2)
