# Deepfake Video Detector

A portfolio-ready research prototype for frame-level deepfake screening. It includes a reproducible EfficientNet-B0 fine-tuning pipeline, face-aware video sampling, and a Streamlit review dashboard.

> **Research prototype:** a model score is not proof that media is authentic or manipulated. This repository does not ship pretrained deepfake weights. Train on a licensed, representative dataset and evaluate across identities, codecs, and generation methods before interpreting results.

## Features

- Fine-tune an ImageNet-initialized EfficientNet-B0 classifier on labeled real/fake frames.
- Sample video frames and score the largest detected face when one is visible.
- Review score trends and aggregate signals in a local web interface.
- Keep large datasets and trained checkpoints outside version control.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Prepare a licensed image dataset with this layout (classes must be named `fake` and `real`):

```text
data/
├── train/
│   ├── fake/
│   └── real/
└── val/
    ├── fake/
    └── real/
```

Train and launch:

```bash
python train.py --data data --epochs 5
streamlit run app.py
```

## Method and limitations

The training script fine-tunes a pretrained EfficientNet-B0 using frame crops. Inference samples every Nth frame, optionally crops the largest Haar-detected face, and reports the mean fake-class probability plus per-sample scores. A temporal average can hide short edits; face detectors can miss profiles or unusual lighting. Test on subject-disjoint and source-disjoint splits, report precision/recall and calibration, and review false positives before any real-world use.

## Data and model provenance

Use only data you have permission to process. Record the source, license, split method, preprocessing, training command, metrics, and checkpoint hash when training a model. Do not commit datasets, videos, or trained weights.

## License

MIT. See [LICENSE](LICENSE).
