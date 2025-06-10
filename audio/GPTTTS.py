#!/usr/bin/env python3
"""
GPTTTS – OpenAI‐TTS helper for AlphaMini or local laptop.

Now supports a `use_robot` flag:
  • if use_robot=True, calls speak_robot(...)
  • if use_robot=False, calls speak(...)

2025-06-05 update:
  • FIX 1 – make PlayAudio blocking (is_serial=True)
  • speak_text() now pauses STT & sets lamp RED before any audio work starts,
    and restores lamp GREEN + resumes STT afterwards.
"""

from __future__ import annotations
import asyncio, functools, logging, os, random, socket, tempfile, threading, wave
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openai.helpers import LocalAudioPlayer
import sounddevice as sd
import soundfile as sf

import mini.mini_sdk as MiniSdk
from mini.apis.api_action     import GetActionList, PlayAction, RobotActionType
from mini.apis.api_sound      import PlayAudio, AudioStorageType
from mini.apis.api_expression import SetMouthLamp, MouthLampMode, MouthLampColor
from mini.dns.dns_browser     import WiFiDevice
from unused.robot_speech_tracker import RobotSpeechTracker

_LOG = logging.getLogger(__name__)

# ─────────── measured speakingAct durations ────────────
SPEAKING_ACT_DURATIONS = {
    'speakingAct1': 1.64,  'speakingAct2': 3.182, 'speakingAct3': 2.176,
    'speakingAct4': 3.20,  'speakingAct5': 3.273, 'speakingAct6': 3.254,
    'speakingAct7': 4.601, 'speakingAct8': 2.329, 'speakingAct9': 2.274,
    'speakingAct10': 2.14, 'speakingAct11': 3.131,'speakingAct12': 3.831,
    'speakingAct13': 1.285,'speakingAct14': 3.57, 'speakingAct15': 3.464,
    'speakingAct16': 3.396,'speakingAct17': 3.028,
}

# ─────────── HTTP helper for streaming WAV ─────────────
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

class _DirHTTPServer:
    def __init__(self, directory: Path | str | None = None):
        self.dir = Path(directory or os.getcwd())
        self.ip  = _lan_ip()
        self.http: HTTPServer | None = None
        self.thr : threading.Thread | None = None
    async def __aenter__(self):
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.dir))
        self.http = HTTPServer(("", 0), handler)
        self.port = self.http.server_address[1]
        self.thr  = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thr.start()
        return f"http://{self.ip}:{self.port}"
    async def __aexit__(self, *exc):
        if self.http: self.http.shutdown()
        if self.thr : self.thr.join()

# ────────────── GPTTTS main class ──────────────────────
class GPTTTS:
    def __init__(
        self,
        api_key: str,
        *,
        use_robot: bool = True,
        robot_serial_suffix: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "coral",
        response_format: str = "pcm",
        default_instructions: str = "Speak in a friendly tone.",
        speech_tracker: Optional[RobotSpeechTracker] = None,
        transcriber=None,
    ) -> None:
        self.use_robot = use_robot
        self.robot_serial_suffix = robot_serial_suffix
        self._connected = False
        self._device: WiFiDevice | None = None

        self.client  = OpenAI(api_key=api_key)
        self.model   = model
        self.voice   = voice
        self.format  = response_format
        self.default_instructions = default_instructions

        self.player  = LocalAudioPlayer()
        self.speech_tracker = speech_tracker
        self.transcriber    = transcriber

        self._idle_lamp: tuple[MouthLampMode, MouthLampColor] | None = None
        self._speak_ids : list[str] = []

    # ──────────── internal helpers ─────────────
    async def _set_mouth_lamp(
        self,
        mode: MouthLampMode,
        color: MouthLampColor,
        duration_ms: int = -1,
        breath_ms: int = 800,
    ):
        await SetMouthLamp(
            mode=mode,
            color=color,
            duration=duration_ms,
            breath_duration=breath_ms,
        ).execute()

    async def _ensure_robot(self, timeout: int = 10):
        if not self.use_robot:
            return
        if self._connected:
            return
        if not self.robot_serial_suffix:
            raise ValueError("robot_serial_suffix not set, but use_robot=True")

        MiniSdk.set_robot_type(MiniSdk.RobotType.EDU)
        dev = await MiniSdk.get_device_by_name(self.robot_serial_suffix, timeout)
        if not dev or not await MiniSdk.connect(dev):
            raise RuntimeError("AlphaMini connection failed")
        await MiniSdk.enter_program()
        self._device    = dev
        self._connected = True
        await asyncio.sleep(2.0)  # finish “enter code mode” prompt

        # cache action IDs w/ durations
        _, resp = await GetActionList(action_type=RobotActionType.INNER).execute()
        self._speak_ids = [
            a.id for a in resp.actionList if a.id in SPEAKING_ACT_DURATIONS
        ]
        _LOG.info("Cached %d speakingAct IDs", len(self._speak_ids))

        # remember current lamp style → treat as “idle”
        self._idle_lamp = (MouthLampMode.BREATH, MouthLampColor.RED)
        await self._set_mouth_lamp(*self._idle_lamp)

    # ──────────── optional explicit warm-up ─────────────
    async def connect_robot(self, *, timeout: int = 10):
        if self.use_robot:
            await self._ensure_robot(timeout)

    # ──────────── local laptop speaker ────────────────
    async def _speak_local(self, text: str):
        """Play TTS locally via sounddevice (laptop mode)."""
        if self.speech_tracker:
            self.speech_tracker.set(text.lower().split())

        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.format,
            instructions=self.default_instructions,
        )
        # write PCM to a temporary WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav = tmp.name
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(resp.content if hasattr(resp, "content") else resp)

        data, samplerate = sf.read(wav)
        sd.play(data, samplerate)
        sd.wait()
        os.unlink(wav)

    # ──────────── robot playback with animation ────────
    async def _speak_robot(self, text: str, *, animate: bool = True):
        await self._ensure_robot()

        # TTS synthesis
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.format,
            instructions=self.default_instructions,
        )
        pcm = resp.content if hasattr(resp, "content") else resp
        tts_dur = len(pcm) / (24000 * 2)

        # write WAV to temp dir
        tmp_dir  = Path(tempfile.mkdtemp())
        wav_path = tmp_dir / "tts.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)

        # stream to robot via temporary HTTP dir
        async with _DirHTTPServer(tmp_dir) as base:
            # ─── FIX 1: use blocking PlayAudio (is_serial=True by default) ───
            play_audio = PlayAudio(
                url=f"{base}/{wav_path.name}",
                storage_type=AudioStorageType.NET_PUBLIC,
                volume=1.0,
            )

            # launch gestures concurrently
            gesture_task = None
            if animate and self._speak_ids:
                gesture_task = asyncio.create_task(self._random_actions(tts_dur))

            await play_audio.execute()   # ← returns *after* audio finishes

            if gesture_task:
                gesture_task.cancel()

        wav_path.unlink(missing_ok=True)
        tmp_dir.rmdir()

    async def _random_actions(self, total_dur: float):
        loop  = asyncio.get_event_loop()
        end_t = loop.time() + total_dur
        while loop.time() < end_t:
            act = random.choice(self._speak_ids)
            dur = SPEAKING_ACT_DURATIONS[act]
            _LOG.debug("Play %s (%.2fs)", act, dur)
            await PlayAction(action_name=act).execute()
            await asyncio.sleep(dur + 0.1)

    # ──────────── unified public entry point ───────────
    async def speak_text(self, text: str, *, animate: bool = True):
        """
        • Pauses transcriber immediately
        • Mouth lamp red while talking
        • Resumes transcriber + lamp green afterwards
        """
        # ----- PRE-SPEECH -----
        if self.transcriber:
            self.transcriber.pause_listening()
        if self.use_robot:
            await self._ensure_robot()
        if self._connected:
            await self._set_mouth_lamp(MouthLampMode.BREATH, MouthLampColor.RED)

        # ----- SPEECH PATH -----
        if self.use_robot:
            await self._speak_robot(text, animate=animate)
        else:
            await self._speak_local(text)

        # ----- POST-SPEECH -----
        if self._connected:
            await self._set_mouth_lamp(MouthLampMode.NORMAL, MouthLampColor.GREEN)
        if self.transcriber:
            self.transcriber.resume_listening()

    # ──────────── shutdown ───────────────
    async def close_robot(self):
        if not self.use_robot or not self._connected:
            return
        await self._set_mouth_lamp(*self._idle_lamp)
        await MiniSdk.quit_program()
        await MiniSdk.release()
        self._connected = False
