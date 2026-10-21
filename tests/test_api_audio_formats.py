import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_api_module():
    # The API module imports the model and text normalizer at startup. Stub them so
    # these format tests never download model weights or process real text/audio.
    chat_tts = types.ModuleType("ChatTTS")
    chat_tts.Chat = lambda: object()
    utils = types.ModuleType("utils")
    utils.batch_split = lambda *args, **kwargs: []
    utils.normalize_zh = lambda text: text
    soundfile = types.ModuleType("soundfile")

    spec = importlib.util.spec_from_file_location("_api_under_test", REPO_ROOT / "api.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "ChatTTS": chat_tts,
        "utils": utils,
        "soundfile": soundfile,
        "_api_under_test": module,
    }):
        spec.loader.exec_module(module)
    return module


class ApiAudioFormatTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = load_api_module()

    async def test_unsupported_format_is_rejected_before_inference(self):
        for media_type in ("mp3", "flac"):
            with self.subTest(media_type=media_type):
                with patch.object(self.api, "generate_tts_audio", side_effect=AssertionError("inference ran")):
                    with self.assertRaises(self.api.HTTPException) as error:
                        await self.api.tts_handle({"media_type": media_type})
                self.assertEqual(error.exception.status_code, 422)

    def test_pack_audio_does_not_label_raw_bytes_as_an_unknown_format(self):
        payload = np.array([1, -2], dtype=np.int16)
        for media_type in ("mp3", "flac"):
            with self.subTest(media_type=media_type):
                with self.assertRaisesRegex(ValueError, "Unsupported media type"):
                    self.api.pack_audio(io.BytesIO(), payload, 24000, media_type)

    def test_aac_encoder_invokes_ffmpeg_and_returns_its_output(self):
        payload = np.array([1, -2], dtype=np.int16)
        process = Mock()
        process.communicate.return_value = (b"synthetic-aac", b"")
        process.returncode = 0
        with patch.object(self.api.subprocess, "Popen", return_value=process):
            encoded = self.api.pack_audio(io.BytesIO(), payload, 24000, "aac")
        self.assertEqual(encoded.getvalue(), b"synthetic-aac")

    def test_aac_encoder_reports_ffmpeg_failure(self):
        payload = np.array([1, -2], dtype=np.int16)
        process = Mock()
        process.communicate.return_value = (b"", b"synthetic error")
        process.returncode = 1
        with patch.object(self.api.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                self.api.pack_audio(io.BytesIO(), payload, 24000, "aac")


if __name__ == "__main__":
    unittest.main()
