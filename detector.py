from pathlib import Path
import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
PREPROCESS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def load_model(checkpoint: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = timm.create_model(saved.get("architecture", "tf_efficientnet_b0"), pretrained=False, num_classes=2)
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    return model, device, saved.get("classes", ["fake", "real"])

def analyze_video(path: Path, bundle, step: int = 12, max_frames: int = 240):
    model, device, classes = bundle
    capture = cv2.VideoCapture(str(path))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rows = []
    frame_index = 0
    while len(rows) < max_frames:
        ok, frame = capture.read()
        if not ok: break
        if frame_index % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
            if len(faces):
                x,y,w,h = max(faces, key=lambda box: box[2]*box[3])
                margin = int(max(w,h)*.12); x0=max(0,x-margin); y0=max(0,y-margin)
                crop = frame[y0:max(y+h+margin,y0+1), x0:max(x+w+margin,x0+1)]
            else:
                crop = frame
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = PREPROCESS(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
            fake_idx = classes.index("fake") if "fake" in classes else 0
            rows.append({"frame": frame_index, "fake_probability": float(probabilities[fake_idx])})
        frame_index += 1
    capture.release()
    return rows
