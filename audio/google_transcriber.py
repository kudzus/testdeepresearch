import itertools
import logging
import pathlib
import queue
import sys
import threading
import time
from queue import Queue
from time import sleep

import grpc
from google.api_core.exceptions import OutOfRange, GoogleAPICallError
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from grpc import StatusCode

# ---------------------------------------------------------------------------
try:
    from unused.robot_speech_tracker import RobotSpeechTracker
except ModuleNotFoundError:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    from unused.robot_speech_tracker import RobotSpeechTracker

from audio.audio_stream import AudioStream

# ---------------------------------------------------------------------------
MAX_RPC_LIFETIME = 240  # seconds (below Google’s 305 s cap)
MUTE_FILL_MS = 100      # silence frame length while muted

# gRPC codes that usually indicate a transient interruption
_RETRYABLE = (
    StatusCode.INTERNAL,            # 13
    StatusCode.UNAVAILABLE,         # 14 – often DNS / network blip
    StatusCode.CANCELLED,           # 1  – server shut connection
    StatusCode.DEADLINE_EXCEEDED,   # 4
)


class GoogleStreamingTranscriber:
    """Bidirectional Google Cloud STT with auto-reconnect and idle timer.

    Public attributes
    -----------------
    last_activity: float  – Unix timestamp of the most recent *interim or
                             final* transcript that contained non-empty text.
    transcription_queue: Queue[str]  – final utterances delivered to caller.
    """

    def __init__(
        self,
        audio_stream: AudioStream,
        language_code: str = "en-US",
        credentials_file: str = "eng-archery-293122-cecac67a97a6.json",
        enable_diarization: bool = True,
        min_speakers: int = 2,
        max_speakers: int = 2,
        speech_tracker: "RobotSpeechTracker | None" = None,
    ):
        # ── audio source ------------------------------------------------------
        self.audio_stream = audio_stream
        self.consumer_id = "gst"
        self.audio_stream.register_consumer(self.consumer_id)

        # ── credentials / client --------------------------------------------
        creds = service_account.Credentials.from_service_account_file(credentials_file)
        self._creds = creds
        self.client = speech.SpeechClient(credentials=self._creds)

        # ── recognition config ----------------------------------------------
        diarization_cfg = (
            speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=min_speakers,
                max_speaker_count=max_speakers,
            )
            if enable_diarization
            else None
        )

        self.rec_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=audio_stream.samplerate,
            language_code=language_code,
            diarization_config=diarization_cfg,
            enable_automatic_punctuation=True,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.rec_config,
            interim_results=True,
            single_utterance=False,
        )

        # ── runtime state ----------------------------------------------------
        self.transcription_queue: "Queue[str]" = Queue()
        self.listening_enabled = True
        self._shutdown = threading.Event()
        self._thread: "threading.Thread | None" = None
        self.last_activity = time.time()

        # diagnostic helper; can be None
        self._speech_tracker = speech_tracker

    # ───────────────── mute / un-mute helpers ──────────────────────────────
    def pause_listening(self):
        """Suspend microphone input (audio is replaced by silence)."""
        self.listening_enabled = False

    def resume_listening(self):
        """Resume microphone input after a short buffer drain."""
        sleep(0.3)
        self.listening_enabled = True

    # ───────────────── audio iterator ──────────────────────────────────────
    def _audio_iterator(self):
        silence = b"\x00" * int(self.audio_stream.samplerate * MUTE_FILL_MS / 1000 * 2)
        for chunk in self.audio_stream.audio_generator(self.consumer_id):
            yield chunk if self.listening_enabled else silence

    # ───────────────── gRPC stream helpers ─────────────────────────────────
    def _open_rpc(self):
        self._rpc_start = time.time()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=b)
            for b in self._audio_iterator()
        )
        return self.client.streaming_recognize(
            requests=requests,
            config=self.streaming_config,
        )

    def _restart_rpc(self):
        # Re-instantiate client to clear internal gRPC state.
        self.client = speech.SpeechClient(credentials=self._creds)
        return self._open_rpc()

    # ───────────────── public thread API ───────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Transcriber already running")
        self._thread = threading.Thread(target=self._transcribe_loop,
                                        name="GoogleSTT", daemon=True)
        self._thread.start()

    def stop(self):
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=2)

    def is_alive(self) -> bool:
        """Return True if the worker thread is still active."""
        return self._thread.is_alive() if self._thread else False

    # ───────────────── main worker loop ────────────────────────────────────
    def _handle_response(self, response):
        if not response.results:
            return
        result = response.results[0]
        text = result.alternatives[0].transcript

        # Idle-timer refresh on ANY interim chunk containing text
        if text.strip():
            self.last_activity = time.time()

        if result.is_final and text.strip():
            self.transcription_queue.put(text.strip())
            if self._speech_tracker:
                self._speech_tracker.record(text)

    def _transcribe_loop(self):
        responses = self._open_rpc()
        backoff = itertools.count(0)  # exponential back-off counter

        while not self._shutdown.is_set():
            try:
                for response in responses:
                    self._handle_response(response)
                    # reset back-off after any successful chunk
                    backoff = itertools.count(0)
                    if self._shutdown.is_set():
                        break

                # Normal stream exhaustion (unlikely). Re-open.
                if not self._shutdown.is_set():
                    logging.info("STT stream ended – reopening")
                    responses = self._restart_rpc()

            except OutOfRange:
                logging.warning("STT OutOfRange – reopening stream")
                responses = self._restart_rpc()

            except grpc.RpcError as err:
                code = err.code()
                if code in _RETRYABLE and not self._shutdown.is_set():
                    delay = min(2 ** next(backoff), 30)
                    logging.warning("STT gRPC %s – retrying in %.1fs", code.name, delay)
                    time.sleep(delay)
                    responses = self._restart_rpc()
                    continue
                logging.exception("Unrecoverable STT gRPC error – stopping")
                break

            except GoogleAPICallError as err:
                logging.exception("Google API error: %s – stopping", err)
                break

        logging.info("STT worker exiting")
