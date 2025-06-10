import threading
import time
from typing import Dict, Optional

from ResidualMaskingNetwork.rmn import RMN

class EmotionDetector:
    def __init__(self, cam_stream, fps: int = 5, method: str = "RMN"):
        self.cam_stream = cam_stream
        self.fps = fps
        self.method = method.upper()
        self._emo_spans: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        if self.method == "RMN":
            self._model = RMN()
        else:
            self._model = None

        self.current_emotion: Optional[str] = None
        self._span_start: float = time.time()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._span_start = time.time()
        self.current_emotion = None
        self._emo_spans.clear()
        self._stop_event.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.fps
        last_ts = 0.0

        while not self._stop_event.is_set():
            frame = self.cam_stream.get_frame()
            if frame is not None:
                now = time.time()
                if now - last_ts >= period:
                    last_ts = now
                    label = self._detect_emotion(frame)

                    if label != self.current_emotion:
                        # Close out the old span
                        if self.current_emotion is not None:
                            duration = round(now - self._span_start, 2)
                            with self._lock:
                                self._emo_spans.append((self.current_emotion, duration))
                            print(f"New emotion detected: {label.capitalize()}")
                            print(f"Previous emotion ('{self.current_emotion}') lasted {duration:.2f} seconds.")
                        else:
                            print(f"New emotion detected: {label.capitalize()}")
                        self.current_emotion = label
                        self._span_start = now
            time.sleep(0.01)

        now = time.time()
        if self.current_emotion is not None:
            duration = round(now - self._span_start, 2)
            with self._lock:
                self._emo_spans.append((self.current_emotion, duration))
            print(f"Detector stopped. Final emotion ('{self.current_emotion}') lasted {duration:.2f} seconds.")

    def _detect_emotion(self, frame) -> str:
        if self.method == "RMN" and self._model is not None:
            dets = self._model.detect_emotion_for_single_frame(frame)
            return dets[0]["emo_label"] if dets else "no-face"
        else:
            return "no-face"

    def get_summary_and_reset(self) -> list[tuple[str, float]]:
        now = time.time()
        with self._lock:
            if self.current_emotion is not None and self._span_start is not None:
                duration = round(now - self._span_start, 2)
                self._emo_spans.append((self.current_emotion, duration))
                self._span_start = now
            summary = list(self._emo_spans)
            self._emo_spans.clear()
        return summary

    def stop(self) -> None:
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._thread.join()
