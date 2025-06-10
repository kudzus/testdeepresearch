# ============================================================================
#  participant_manager.py  – handles per‑participant storage
# ============================================================================
"""
Disk layout
───────────
participants/<PID>/
    ├── video_001/clip.mp4
    ├── video_002/clip.mp4
    ├── emotion_log.csv   (turn_idx, timestamp_iso, emotion, duration_s)
    ├── conversation.jsonl
    └── timeline.log      (timestamp_iso  |  free‑text marker)
"""

import csv, json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ParticipantDataManager:
    """Owns all disk IO for *one* participant ID."""

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, participant_id: str, base_dir: str = "participants"):
        self.participant_id = participant_id.strip()
        self.part_dir       = Path(base_dir) / self.participant_id
        self.part_dir.mkdir(parents=True, exist_ok=True)

        # ---- video clip numbering --------------------------------------
        existing = sorted(d for d in self.part_dir.glob("video_*") if d.is_dir())
        self.next_video_idx = len(existing) + 1

        # ---- file handles ----------------------------------------------
        self._emotion_csv_path = self.part_dir / "emotion_log.csv"
        self._chat_jsonl_path  = self.part_dir / "conversation.jsonl"
        self._timeline_path    = self.part_dir / "timeline.log"

        # create header rows for brand‑new participant -------------------
        if not self._emotion_csv_path.exists():
            with open(self._emotion_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["turn_idx", "timestamp_iso", "emotion", "duration_s"]
                )
        if not self._timeline_path.exists():
            self._timeline_path.touch()

    # ─────────────────────────────────────────────────────────────────────
    # VIDEO
    # ─────────────────────────────────────────────────────────────────────
    def _new_video_folder(self) -> Path:
        folder = self.part_dir / f"video_{self.next_video_idx:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        self.next_video_idx += 1
        return folder

    def record_and_save_clip(self, clip_recorder, duration: float = 5.0) -> Path:
        """Capture a short .mp4 from the shared camera stream."""
        vid_dir = self._new_video_folder()
        return clip_recorder.record_and_save(duration=duration,
                                             filename="clip.mp4",
                                             dest_dir=vid_dir)

    # ─────────────────────────────────────────────────────────────────────
    # EMOTION LOGGING
    # ─────────────────────────────────────────────────────────────────────
    def append_emotion_summary(self, turn_idx: int, emo_durations: Dict[str, float]):
        """Append one row per (emotion, duration) pair."""
        ts_iso = datetime.utcnow().isoformat()
        with open(self._emotion_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for emo, dur in emo_durations.items():
                w.writerow([turn_idx, ts_iso, emo, round(dur, 3)])

    # ─────────────────────────────────────────────────────────────────────
    # CONVERSATION LOGGING (system prompt + assistant)
    # ─────────────────────────────────────────────────────────────────────
    def append_chat_turn(
        self,
        turn_idx: int,
        system_prompt: str,
        assistant_response: str,
        extra: Dict[str, Any] | None = None,
    ):
        record: Dict[str, Any] = {
            "turn_idx": turn_idx,
            "timestamp_iso": datetime.utcnow().isoformat(),
            "system_prompt": system_prompt,
            "assistant_response": assistant_response,
        }
        if extra:
            record.update(extra)
        with open(self._chat_jsonl_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    # ─────────────────────────────────────────────────────────────────────
    # TIMELINE LOGGING (free‑text markers)
    # ─────────────────────────────────────────────────────────────────────
    def append_timeline(self, message: str):
        ts_iso = datetime.utcnow().isoformat()
        with open(self._timeline_path, "a", encoding="utf-8") as f:
            f.write(f"{ts_iso}\t{message}\n")
