from pathlib import Path
import tempfile
import streamlit as st
from detector import analyze_video, load_model

ROOT = Path(__file__).parent
st.set_page_config(page_title="Deepfake Lens", page_icon="◉", layout="wide")
st.title("◉ Deepfake Lens")
st.caption("Frame-level video review powered by a project-trained EfficientNet model.")
model_path = ROOT / "models" / "best.pt"
if not model_path.exists():
    st.warning("No trained model is installed yet. Follow the training guide to create models/best.pt before analyzing videos.")
    st.stop()
model = load_model(model_path)
video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"] )
step = st.slider("Sample every Nth frame", 1, 60, 12)
if video and st.button("Analyze video", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as f:
        f.write(video.getvalue())
        temp_path = Path(f.name)
    try:
        with st.spinner("Sampling frames and scoring the video..."):
            results = analyze_video(temp_path, model, step=step)
        if not results:
            st.error("No usable frames were found in this video.")
        else:
            fake_scores = [r["fake_probability"] for r in results]
            mean_score = sum(fake_scores) / len(fake_scores)
            a, b, c = st.columns(3)
            a.metric("Frames analyzed", len(results))
            b.metric("Mean fake-class score", f"{mean_score:.1%}")
            c.metric("Review label", "Elevated" if mean_score >= .5 else "Lower")
            st.line_chart({"Fake-class score": fake_scores})
            st.caption("This is a model score, not proof of manipulation. Validate against representative data and inspect source media before drawing conclusions.")
    finally:
        temp_path.unlink(missing_ok=True)
