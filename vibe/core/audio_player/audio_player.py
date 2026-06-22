from __future__ import annotations

from collections.abc import Callable
import threading
from typing import TYPE_CHECKING, Any

from vibe.core.audio_player.audio_player_port import (
    AlreadyPlayingError,
    AudioBackendUnavailableError,
    AudioFormat,
    NoAudioOutputDeviceError,
    UnsupportedAudioFormatError,
)
from vibe.core.audio_player.utils import decode_wav
from vibe.core.logger import logger

if TYPE_CHECKING:
    import sounddevice as sd
    from sounddevice import CallbackFlags, RawOutputStream


# sounddevice loads a native PortAudio library at import (and enumerates audio
# devices), which is slow and unnecessary when voice features are unused. Import
# it lazily on first use instead of at module import. It raises OSError when no
# audio driver is available, in which case `sd` is None and playback is disabled.
def _load_sounddevice() -> None:
    """Populate the module-global ``sd`` once (real module, or None on OSError)."""
    if "sd" not in globals():
        try:
            import sounddevice

            globals()["sd"] = sounddevice
        except OSError:
            globals()["sd"] = None


def __getattr__(name: str) -> Any:
    # PEP 562: lets external access (incl. tests patching `...audio_player.sd`)
    # trigger the lazy load, so `sd` behaves like a normal module attribute.
    if name == "sd":
        _load_sounddevice()
        return globals()["sd"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


DEFAULT_BLOCKSIZE = 4096
DTYPE = "int16"
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


class AudioPlayer:
    """Plays audio through the default output device using sounddevice."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: RawOutputStream | None = None
        self._playing: bool = False
        self._audio_data: bytes = b""
        self._position: int = 0
        self._frame_size: int = 0
        self._on_finished: Callable[[], object] | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(
        self,
        audio_data: bytes,
        audio_format: AudioFormat,
        *,
        on_finished: Callable[[], object] | None = None,
    ) -> None:
        _load_sounddevice()
        with self._lock:
            if self._playing:
                raise AlreadyPlayingError("Already playing")

            if not sd:
                error_message = "sounddevice is not available, audio playback disabled"
                logger.error(error_message)
                raise AudioBackendUnavailableError(error_message)

            self._guard_audio_output()

            match audio_format:
                case AudioFormat.WAV:
                    sample_rate, channels, pcm_data = decode_wav(audio_data)
                case _:
                    raise UnsupportedAudioFormatError(
                        f"Unsupported audio format: {audio_format}"
                    )
            self._audio_data = pcm_data
            self._position = 0
            self._frame_size = channels * DEFAULT_SAMPLE_WIDTH
            self._on_finished = on_finished

            self._stream = sd.RawOutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=DTYPE,
                blocksize=DEFAULT_BLOCKSIZE,
                callback=self._audio_callback,
                finished_callback=self._on_stream_finished,
            )
            self._stream.start()
            self._playing = True

    def stop(self) -> None:
        stream = self._stream
        if not self._playing or stream is None:
            return
        stream.close(ignore_errors=True)

    def _audio_callback(
        self, outdata: memoryview, frames: int, time_info: object, status: CallbackFlags
    ) -> None:
        _load_sounddevice()
        if not sd:
            raise RuntimeError("sounddevice is not available")
        if status:
            logger.warning("Audio playback callback status: %s", status)

        bytes_needed = frames * self._frame_size
        chunk = self._audio_data[self._position : self._position + bytes_needed]
        self._position += len(chunk)

        if len(chunk) < bytes_needed:
            outdata[: len(chunk)] = chunk
            outdata[len(chunk) :] = b"\x00" * (bytes_needed - len(chunk))
            raise sd.CallbackStop()
        else:
            outdata[:] = chunk

    def _on_stream_finished(self) -> None:
        on_finished = None
        with self._lock:
            self._stream = None
            self._playing = False
            on_finished = self._on_finished

        if on_finished is not None:
            on_finished()

    @staticmethod
    def _guard_audio_output() -> None:
        _load_sounddevice()
        if sd is None:
            raise RuntimeError("sounddevice is not available")
        try:
            sd.query_devices(kind="output")
        except Exception as exc:
            raise NoAudioOutputDeviceError("No audio output device available") from exc
