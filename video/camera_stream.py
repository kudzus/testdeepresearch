# video/camera_stream.py

import cv2
import time
import threading
from collections import deque

class CameraStream:
    def __init__(self, camera_index=0, buffer_size=30):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")

        # ── Circular buffer for “recent frames” ─────────────────────────────
        self.frame_buffer = deque(maxlen=buffer_size)
        self.latest_frame = None

        # ── LISTENERS: each is a callable(frame) to be invoked on every new frame
        self.listeners = []

        # ── Thread control ─────────────────────────────────────────────────
        self.stopped = False
        self.lock = threading.Lock()

        # ── Start background capture thread ────────────────────────────────
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        Continuously capture frames, store in circular buffer,
        and dispatch each new frame to all registered listeners.
        """
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.lock:
                self.latest_frame = frame
                self.frame_buffer.append(frame)
                listeners_copy = list(self.listeners)

            # Notify each listener *without* holding the lock
            for callback in listeners_copy:
                try:
                    # We pass the raw frame; listeners must copy if they need to mutate it.
                    callback(frame)
                except Exception:
                    # If a listener throws, ignore it—don’t break the loop.
                    pass

            time.sleep(0.01)  # avoid spinning at 100% CPU

    def get_frame(self):
        """Return a copy of the most recent frame (or None if none yet)."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_frames(self):
        """Return copies of all frames currently in the circular buffer."""
        with self.lock:
            return [f.copy() for f in self.frame_buffer]

    def add_frame_listener(self, callback):
        """
        Register a callable `callback(frame)` that will be invoked on every new frame.
        """
        with self.lock:
            self.listeners.append(callback)

    def remove_frame_listener(self, callback):
        """Unregister a previously‐added callback."""
        with self.lock:
            if callback in self.listeners:
                self.listeners.remove(callback)

    def stop(self):
        """Stop the capture thread and release camera resources."""
        self.stopped = True
        self.thread.join()
        self.cap.release()
