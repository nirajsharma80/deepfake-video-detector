# Deepfake Lens

Deepfake Lens is a local Streamlit app for Gemini-assisted screening of photos and videos. It reports qualitative cues and uncertainty; it is not a forensic detector and cannot prove that media is authentic or manipulated.

## Features

- Analyze JPG, JPEG, PNG, and WebP images.
- Analyze MP4, MOV, AVI, WebM, WMV, MPEG, MPG, 3GP, and FLV videos.
- Choose Gemini 3.5 Flash-Lite for faster replies or Gemini 3.8 Flash for a stronger review.
- See the model's assessment, qualitative confidence, cited visual/audible cues, and limitations.
- Require an explicit in-app consent checkbox before sending selected media to Google.
- Request deletion of uploaded video files after analysis; the Gemini Files API also expires uploaded files after 48 hours.
- Keep the earlier EfficientNet training code as an optional local research path.

## Run on Windows with VS Code

Use Python 3.11–3.15. In the VS Code terminal, from this folder, run:

```powershell
python -m venv .venv-vscode
.\.venv-vscode\Scripts\python.exe -m pip install --upgrade pip
.\.venv-vscode\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in VS Code and set your Gemini API key:

```text
GOOGLE_API_KEY=your_key_here
```

For a one-session run, you can instead enter the key in the app's masked sidebar field.

Then launch the app:

```powershell
.\.venv-vscode\Scripts\python.exe -m streamlit run app.py
```

Open the local URL Streamlit prints, usually <http://localhost:8501>. `.env` is ignored by Git; never commit it or share its key. You can also set `GOOGLE_API_KEY` in the same terminal session before launching instead of using `.env`.

## Media handling and limitations

Photo bytes are sent inline in the Gemini request. Videos are uploaded through the Gemini Files API for processing, then Deepfake Lens requests deletion after analysis. The app shows a warning and requires consent before either request. Avoid confidential or sensitive media. Gemini API usage may be billable under your account.

Gemini is a general multimodal model, not a purpose-trained deepfake classifier. It can miss subtle edits, be misled by compression or unusual lighting, and produce false positives. Its confidence wording is qualitative and uncalibrated. Use source provenance, forensic tooling, and human review for consequential decisions.

## Optional local training pipeline

The original EfficientNet-B0 training and local video inference code remains available separately. It requires its own compatible PyTorch environment and a licensed dataset arranged as:

```text
data/
├── train/
│   ├── fake/
│   └── real/
└── val/
    ├── fake/
    └── real/
```

Install `requirements-training.txt`, then run `python train.py --data data --epochs 5`. The Streamlit app does not require this dataset or a checkpoint.

## License

MIT. See [LICENSE](LICENSE).
