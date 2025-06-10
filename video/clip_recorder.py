# video/clip_recorder.py

import os
import time
import cv2

class ClipRecorder:
    """
    Collect the next N seconds of frames from an existing CameraStream and
    (optionally) save them to disk.
    """
    def __init__(self, camera_stream, clips_dir="saved_clips"):
        """
        :param camera_stream: an existing CameraStream instance.
        :param clips_dir: relative path (within repo) where clips will be saved.
                          Defaults to "saved_clips". Directory will be created if needed.
        """
        self.camera    = camera_stream
        self.clips_dir = clips_dir
        os.makedirs(self.clips_dir, exist_ok=True)

    def record_clip(self, duration=5.0):
        """
        Record the next `duration` seconds of video and return a list of frames.

        Mechanism:
          1) Create a temporary list `collected = []`.
          2) Define a local callback that, on each new frame, does
             `collected.append(frame.copy())`.
          3) Register that callback with `camera.add_frame_listener(...)`.
          4) Sleep for `duration` seconds.
          5) Un‐register the callback and return the list of collected frames.

        :param duration: number of seconds to record (float or int).
        :return: list of NumPy‐arrays (BGR frames) in chronological order.
        """
        collected = []

        def _on_new_frame(frame):
            # Always store a copy, so further processing can’t mutate it.
            collected.append(frame.copy())

        # (1) Register listener
        self.camera.add_frame_listener(_on_new_frame)

        # (2) Wait for exactly `duration` seconds
        time.sleep(duration)

        # (3) Unregister listener
        self.camera.remove_frame_listener(_on_new_frame)

        # (4) Return the collected frames
        return collected

    def save_frames_as_video(self, frames, filename, fps=30, dest_dir=None):
        """
        Save `frames` to <dest_dir>/<filename>.mp4 (defaults to `self.clips_dir`).

        :param frames: list of NumPy‐arrays (BGR frames) to write.
        :param filename: filename (e.g. "clip.mp4") under dest_dir or clips_dir.
        :param fps: frames per second for the output video. Default is 30.
        :param dest_dir: optional override directory; if None, uses `self.clips_dir`.
        :return: full path to the saved video.
        """
        if not frames:
            raise ValueError("No frames to save.")

        base_dir = dest_dir or self.clips_dir
        os.makedirs(base_dir, exist_ok=True)

        # Ensure filename has a video extension
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            filename += ".mp4"

        output_path = os.path.join(base_dir, filename)

        # Grab dimensions from the first frame
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()

        return output_path

    def record_and_save(self, duration=5.0, filename=None, fps=30, dest_dir=None):
        """
        Record the next `duration` seconds of video and immediately save to disk.

        :param duration: number of seconds to record.
        :param filename: desired filename (without path). If None, auto‐generate a timestamped name.
        :param fps: frames per second for the saved video.
        :param dest_dir: optional override directory for saving; if None, uses `self.clips_dir`.
        :return: full path to the saved video.
        """
        # 1) Record frames
        frames = self.record_clip(duration=duration)

        # 2) Determine filename
        if filename is None:
            timestamp = int(time.time())
            filename = f"clip_{timestamp}.mp4"

        # 3) Save and return path
        return self.save_frames_as_video(frames, filename, fps=fps, dest_dir=dest_dir)
