from __future__ import annotations

from typing import get_args


def test_local_speech_output_format_matches_mistralai_sdk() -> None:
    from mistralai.client.models import SpeechOutputFormat as SDKSpeechOutputFormat

    from vibe.core.config.models import SpeechOutputFormat

    assert set(get_args(SpeechOutputFormat)) == set(get_args(SDKSpeechOutputFormat))
