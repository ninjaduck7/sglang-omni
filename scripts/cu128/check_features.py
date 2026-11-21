"""Probe the active local speech server, using the two qualified Vivian clips."""

import argparse
import io
import unicodedata
import wave
from pathlib import Path

import requests


def words(text):
    return "".join(c.casefold() for c in text if unicodedata.category(c)[0] in "LN")


def check_asr(client, url, clips):
    cases = [
        ("en", None, "verbose_json", "English"),
        ("zh", "ZH-cn", "verbose_json", "Chinese"),
        ("en", "invalid-language", "json", None),
        ("zh", None, "verbose_json", "Chinese"),
        ("en", "English", "json", None),
    ]
    expected = {"en": "The weather is sunny today.", "zh": "今天天气很好。"}
    for clip, hint, fmt, language in cases:
        params = {"response_format": fmt}
        if hint:
            params["language"] = hint
        with (clips / f"vivian-{clip}.wav").open("rb") as audio:
            r = client.post(url + "/transcriptions", data=params,
                            files={"file": ("clip.wav", audio, "audio/wav")}, timeout=60)
        if hint == "invalid-language":
            assert r.status_code == 400 and "Unsupported language:" in r.text, r.text
        else:
            r.raise_for_status()
            body = r.json()
            assert words(body["text"]) == words(expected[clip]), body
            if fmt == "json":
                assert set(body) == {"text"}, body
            else:
                assert body["language"] == language, body
        print(clip, hint, fmt, r.status_code, r.text, flush=True)


def check_tts(client, url):
    for budget, reason in [(4, "length"), (128, "stop")]:
        r = client.post(url + "/speech", json={
            "input": "The weather is sunny today.", "voice": "Vivian",
            "language": "English", "max_new_tokens": budget, "response_format": "wav",
        }, timeout=90)
        r.raise_for_status()
        assert r.headers.get("X-Finish-Reason") == reason, r.headers
        tokens = int(r.headers["X-Completion-Tokens"])
        assert (tokens == budget if reason == "length" else 0 < tokens < budget), tokens
        with wave.open(io.BytesIO(r.content)) as wav:
            assert wav.getframerate() == 24000 and wav.getnframes() > 0
            print(reason, tokens, wav.getnframes(), "frames", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["asr", "tts"])
    p.add_argument("--port", type=int, default=18591)
    p.add_argument("--clips", type=Path, default=Path("../../runs/sglang-omni/cu128-5090"))
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/audio"
    with requests.Session() as client:
        client.trust_env = False
        if args.task == "asr":
            check_asr(client, url, args.clips)
        else:
            check_tts(client, url)
