import os
import tempfile
from pathlib import Path

import streamlit as st
from google.genai import errors

from gemini_detector import analyze_image, analyze_video

ROOT = Path(__file__).parent
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".3gp": "video/3gpp",
    ".flv": "video/x-flv",
}

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT / ".env", override=False)
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

st.set_page_config(page_title="Deepfake Lens", page_icon="◉", layout="wide")
st.title("◉ Deepfake Lens")
st.caption("Gemini-assisted screening for possible image and video manipulation.")

st.warning(
    "This is an AI screening aid, not forensic verification. Gemini may miss edits or flag "
    "authentic media. Selected media is sent to Google for analysis; avoid confidential content. "
    "API usage may incur charges.",
    icon="⚠️",
)

with st.sidebar:
    st.subheader("Analysis settings")
    model_name = st.selectbox(
        "Gemini model",
        options=("gemini-3.5-flash-lite", "gemini-3.8-flash"),
        key="deepfake_gemini_model_v1",
    )
    if API_KEY:
        st.caption("Gemini API key loaded.")
    else:
        API_KEY = st.text_input(
            "Google Gemini API key",
            type="password",
            key="gemini_api_key_input",
            help="This fallback stays in the current app session. For repeat use, set GOOGLE_API_KEY in .env.",
        ).strip()

consent = st.checkbox(
    "I understand that the selected media will be sent to Google Gemini for analysis.",
    key="media_upload_consent",
)


def _show_api_error(exc: errors.APIError) -> None:
    if exc.code == 400 or exc.code == 403:
        st.error("Google rejected the request. Check that the API key is valid and Gemini API is enabled.")
    elif exc.code == 404:
        st.error(f"The selected Gemini model is unavailable: {model_name}.")
    elif exc.code == 429:
        st.error("Gemini's request limit was reached. Wait a little and try again.")
    elif exc.code == 503:
        st.error("Gemini is temporarily busy. Wait a moment and try again.")
    else:
        st.error(f"Gemini request failed (HTTP {exc.code}): {exc.message}")


def _check_before_analysis(upload, maximum_size: int) -> bool:
    if not consent:
        st.error("Confirm the media-sharing notice before starting analysis.")
        return False
    if not API_KEY:
        st.error("GOOGLE_API_KEY is not configured. Add it to this project's .env file and restart.")
        return False
    if upload.size > maximum_size:
        st.error(f"This file is too large. The limit is {maximum_size // (1024 * 1024)} MB.")
        return False
    return True


def _display_result(result: str) -> None:
    if not result.strip():
        st.error("Gemini returned no analysis. Try another supported file or model.")
        return
    st.markdown("### Gemini assessment")
    st.markdown(result)
    st.caption(
        "Treat this assessment as a screening signal only. Confirm with provenance, source records, "
        "and specialist forensic tools before making consequential decisions."
    )


photo_tab, video_tab = st.tabs(["Photo detection", "Video detection"])

with photo_tab:
    photo = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=15,
        key="photo_upload",
    )
    if photo:
        st.image(photo, caption=photo.name, use_container_width=True)
        if st.button("Analyze photo", type="primary", key="analyze_photo"):
            if _check_before_analysis(photo, MAX_IMAGE_BYTES):
                try:
                    with st.spinner("Reviewing the photo with Gemini..."):
                        result = analyze_image(
                            data=photo.getvalue(),
                            mime_type=photo.type or "image/jpeg",
                            model=model_name,
                            api_key=API_KEY,
                        )
                    _display_result(result)
                except errors.APIError as exc:
                    _show_api_error(exc)
                except Exception as exc:
                    st.error(f"Photo analysis failed: {exc}")

with video_tab:
    video = st.file_uploader(
        "Choose a video",
        type=list(VIDEO_MIME_TYPES),
        max_upload_size=100,
        key="video_upload",
    )
    if video:
        st.video(video)
        if st.button("Analyze video", type="primary", key="analyze_video"):
            if _check_before_analysis(video, MAX_VIDEO_BYTES):
                suffix = Path(video.name).suffix.lower()
                mime_type = VIDEO_MIME_TYPES.get(suffix, "video/mp4")
                try:
                    with tempfile.TemporaryDirectory(prefix="deepfake-lens-") as temp_dir:
                        video_path = Path(temp_dir) / f"upload{suffix}"
                        video_path.write_bytes(video.getvalue())
                        with st.spinner("Uploading and reviewing the video with Gemini..."):
                            result, file_deleted = analyze_video(
                                path=video_path,
                                mime_type=mime_type,
                                model=model_name,
                                api_key=API_KEY,
                            )
                    _display_result(result)
                    if not file_deleted:
                        st.warning(
                            "Google did not confirm deletion of the temporary video upload. "
                            "The Gemini Files API automatically removes uploaded files after 48 hours."
                        )
                except errors.APIError as exc:
                    _show_api_error(exc)
                except Exception as exc:
                    st.error(f"Video analysis failed: {exc}")
