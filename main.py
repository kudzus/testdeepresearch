# =============================================================
#  main.py  – crossword robot with emotion & timeline logging
#  (June 2025 update 2 – add emotion summaries into timeline.log)
# =============================================================
#!/usr/bin/env python3
"""
main.py – crossword robot with idle chat + emotion context +
          rich on-disk logging.

Update (6 Jun 2025)
───────────────────
• Each time we capture an emotion summary we now ALSO record a plain-text
  entry in *timeline.log* so the temporal flow is visible in one file.
  Examples lines:
      2025-06-06T10:12:03Z  Emotion summary (intro): {"neutral": 18.3}
      2025-06-06T10:12:30Z  Emotion summary after user: {…}

Folder layout for a participant
participants/<PID>/
    ├── video_001/clip.mp4
    ├── emotion_log.csv        ← CSV (turn_idx, emotion, duration)
    ├── conversation.jsonl     ← JSONL (system_prompt, assistant)
    └── timeline.log           ← free-text chronological markers *+ emotions*
"""

import os, logging, time, cv2, queue, asyncio, json
from collections import defaultdict
from threading import Thread
from pathlib import Path
from typing import Dict, List, Tuple, Any

from openai import OpenAI

# ── project imports ───────────────────────────────────────────────────────────
from app import run as run_flask, game_state, socketio, state_ready, get_server_links
from video.camera_stream import CameraStream
from video.emotion_detector import EmotionDetector
from audio.audio_stream import AudioStream
from audio.google_transcriber import GoogleStreamingTranscriber
from audio.GPTTTS import GPTTTS
from config.apikeys import OPENAI_API_KEY
from crossword_prompt import create_system_prompt, _CLUE_LOOKUP
from video.clip_recorder import ClipRecorder
from participant_manager import ParticipantDataManager

# ── idle / emotion parameters ────────────────────────────────────────────────
IDLE_TIMEOUT  = 20          # seconds with no speech
STT_POLL_SECS = 0.5         # Google queue poll interval
FPS_ANALYSE   = 5           # emotion FPS

# ──────────────────────────────────────────────────────────────────────────────

def _drain_transcription_queue(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

def _aggregate_emotions(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for emo, dur in pairs:
        totals[emo] += dur
    return dict(totals)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('rmn').setLevel(logging.WARNING)

    Thread(target=run_flask, daemon=True).start()
    print("[MAIN] Flask-SocketIO server started …")

    cam_stream      = CameraStream(camera_index=0, buffer_size=30)
    emotion_detector = EmotionDetector(cam_stream, fps=FPS_ANALYSE)
    emotion_detector.start()

    clip_recorder = ClipRecorder(cam_stream)

    audio_stream = AudioStream(samplerate=16000, channels=1,
                               dtype="int16", buffer_duration=10.0)
    transcriber  = GoogleStreamingTranscriber(audio_stream, "en-US")
    transcriber.start()

    USE_ROBOT  = True
    tts_manager = GPTTTS(api_key=OPENAI_API_KEY,
                         use_robot=USE_ROBOT,
                         robot_serial_suffix="00233",
                         transcriber=transcriber)

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    transcriber.pause_listening()
    _loop.run_until_complete(tts_manager.connect_robot())
    links = get_server_links()
    print(f"[SETUP] Connect your browser at: {links}")

    participant_id = input("\nParticipant number (anon. ID) ➜ ").strip()
    pdata = ParticipantDataManager(participant_id)
    print(f"[MAIN] created data folder: {pdata.part_dir}")

    print("\n[SETUP] You have 30 s to get ready…")
    for i in range(5, 0, -1):
        print(f"   starting in {i:2d} s ⏳", end="\r", flush=True)
        time.sleep(1)
    print("\n[SETUP] Time’s up – experiment begins!\n")

    transcriber.resume_listening()
    _drain_transcription_queue(transcriber.transcription_queue)
    transcriber.last_activity = time.time()

    history: List[Dict[str, str]] = []
    prev_idle_log = -1
    turn_idx      = 0

    # ─── intro ────────────────────────────────────────────────────────────
    intro_text = (
        "Hey there! I’m Lexi—your friendly crossword side-kick. "
        "I can see which clue you’re working on and I’ll jump in with hints, "
        "fun facts, or just a bit of banter whenever you like. "
        "If my mouth lamp is glowing green, I’m listening! "
        "Ready when you are—good luck, and let’s crack this puzzle together!"
    )

    pdata.append_timeline("Robot starts speaking (intro)")
    _loop.run_until_complete(tts_manager.speak_text(intro_text))

    emo_intro = _aggregate_emotions(emotion_detector.get_summary_and_reset())
    pdata.append_emotion_summary(turn_idx, emo_intro)
    pdata.append_timeline(f"Robot done speaking (intro) said \"{intro_text}\"")
    pdata.append_timeline(f"Emotion summary (intro): {json.dumps(emo_intro, ensure_ascii=False)}")

    history.append({"role": "assistant", "content": intro_text})

    completed_set: set[Tuple[str, int]] = set()

    try:
        while True:
            if not transcriber.is_alive():
                logging.warning("STT thread died – spinning up a new one.")
                transcriber = GoogleStreamingTranscriber(audio_stream, "en-US")
                transcriber.start()

            idle_sec = int(time.time() - transcriber.last_activity)
            if idle_sec != prev_idle_log:
                prev_idle_log = idle_sec
                print(f"[IDLE] {idle_sec:2d}s since last speech", end="\r")

            try:
                user_input = transcriber.transcription_queue.get(timeout=STT_POLL_SECS)
            except queue.Empty:
                user_input = None

            if user_input is None and idle_sec >= IDLE_TIMEOUT:
                print()
                print("[MAIN] idle → injecting [[IDLE]]")
                user_input = "[[IDLE]]"
                transcriber.last_activity = time.time()

            if user_input is None:
                continue
            if user_input.lower().strip() == "quit":
                break

            # USER turn done ---------------------------------------------------
            turn_idx += 1

            emo_after_user = _aggregate_emotions(emotion_detector.get_summary_and_reset())
            pdata.append_emotion_summary(turn_idx, emo_after_user)
            pdata.append_timeline(f"User done speaking said \"{user_input}\"")
            pdata.append_timeline(f"Emotion summary after user: {json.dumps(emo_after_user, ensure_ascii=False)}")

            state_ready.clear()
            socketio.emit("request_state")
            if not state_ready.wait(timeout=0.4):
                print("[MAIN] ⚠️ snapshot timeout – stale state")

            crossword_state = game_state.serialize()

            recently_completed: List[Tuple[str, int]] = []
            for kdir in ("across", "down"):
                for num_str, pat in crossword_state.get(kdir, {}).items():
                    if num_str == "undefined" or not pat or "0" in pat:
                        continue
                    num = int(num_str)
                    dir_letter = kdir[0].upper()
                    answer = _CLUE_LOOKUP[(dir_letter, num)]["answer"]  # type: ignore
                    if pat == answer and (dir_letter, num) not in completed_set:
                        completed_set.add((dir_letter, num))
                        recently_completed.append((dir_letter, num))

            try:
                if not crossword_state:
                    raise KeyError
                system_msg_text = create_system_prompt(
                    game_state=crossword_state,
                    silence_seconds=idle_sec,
                    idle_threshold=IDLE_TIMEOUT,
                    recently_completed=recently_completed,
                )
            except KeyError:
                links = get_server_links()
                print("\n[WARNING] Crossword not open. Connect your browser at:")
                for l in links:
                    print("   •", l)
                system_msg_text = "### ROLE\ncrossword puzzle not yet connected."

            messages = (
                [{"role": "developer", "content": system_msg_text}] +
                history +
                [{"role": "user", "content": user_input}]
            )
            client = OpenAI(api_key=OPENAI_API_KEY)
            assistant_text = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=messages,
            ).choices[0].message.content

            pdata.append_chat_turn(turn_idx, system_msg_text, assistant_text,
                                   extra={"user_input": user_input})

            history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_text},
            ])

            # ROBOT speaks ------------------------------------------------------
            pdata.append_timeline("Robot starts speaking")
            transcriber.pause_listening()
            _loop.run_until_complete(tts_manager.speak_text(assistant_text))
            transcriber.resume_listening()

            emo_after_robot = _aggregate_emotions(emotion_detector.get_summary_and_reset())
            pdata.append_emotion_summary(turn_idx, emo_after_robot)
            pdata.append_timeline(f"Robot done speaking said \"{assistant_text}\"")
            pdata.append_timeline(f"Emotion summary after robot: {json.dumps(emo_after_robot, ensure_ascii=False)}")

            _drain_transcription_queue(transcriber.transcription_queue)
            transcriber.last_activity = time.time()
            prev_idle_log = -1

            def _save_clip():
                video_path = pdata.record_and_save_clip(clip_recorder, duration=5.0)
                print(f"[DATA] clip saved → {video_path}")
            Thread(target=_save_clip, daemon=True).start()

    except KeyboardInterrupt:
        print("\n[MAIN] exiting …")

    finally:
        cam_stream.stop()
        audio_stream.stop()
        if USE_ROBOT:
            _loop.run_until_complete(tts_manager.close_robot())
        _loop.close()
        cv2.destroyAllWindows()


# ============================================================================
#  participant_manager.py – identical to previous version (unchanged)
# ============================================================================
