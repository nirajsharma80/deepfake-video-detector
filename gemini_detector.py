from pathlib import Path
import time

from google import genai
from google.genai import types

ANALYSIS_PROMPT = """Review this media for visible or audible signs consistent with synthetic or manipulated content.

Return a concise Markdown report with these headings:
### Assessment
Choose exactly one: Likely manipulated, No clear manipulation visible, or Inconclusive.
### Confidence
Choose Low, Medium, or High. This is a qualitative impression, not a calibrated probability.
### Evidence
List only specific cues actually present in this media. If none are apparent, say so.
### Limitations
Explain what this file and analysis cannot establish.

Look for relevant visual inconsistencies such as facial texture or boundary artifacts, lighting or
reflection mismatches, unnatural geometry, and frame-to-frame flicker. For video, note that sampling
or compression can hide important details. Do not infer authenticity from realism, and do not claim
to know the media's origin. When evidence is weak or ambiguous, choose Inconclusive. This is a
screening aid, not forensic proof."""


def _client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=180_000),
    )


def _response_text(response) -> str:
    text = response.text
    if not text:
        raise RuntimeError("Gemini did not return readable analysis text.")
    return text.strip()


def analyze_image(data: bytes, mime_type: str, model: str, api_key: str) -> str:
    """Analyze a photo inline; image bytes are not stored with the Gemini Files API."""
    client = _client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            ANALYSIS_PROMPT,
            types.Part.from_bytes(data=data, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=900,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return _response_text(response)


def analyze_video(
    path: Path,
    mime_type: str,
    model: str,
    api_key: str,
    processing_timeout: int = 180,
) -> tuple[str, bool]:
    """Upload and analyze a video, then request deletion of its temporary Google file."""
    client = _client(api_key)
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type=mime_type),
    )
    result = ""
    file_deleted = False

    try:
        deadline = time.monotonic() + processing_timeout
        while True:
            state = getattr(uploaded.state, "name", str(uploaded.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Google could not process this video file.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Google video processing did not finish within 3 minutes.")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        response = client.models.generate_content(
            model=model,
            contents=[
                ANALYSIS_PROMPT,
                types.Part.from_uri(
                    file_uri=uploaded.uri,
                    mime_type=uploaded.mime_type or mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        result = _response_text(response)
    finally:
        try:
            client.files.delete(name=uploaded.name)
            file_deleted = True
        except Exception:
            file_deleted = False

    return result, file_deleted
