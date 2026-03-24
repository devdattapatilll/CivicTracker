"""
COLAB TRAINING GUIDE — CivicTrack 4-Model YOLO + NLP Training
================================================================
Run this in Google Colab with free T4 GPU:
  Runtime → Change runtime type → T4 GPU

Copy-paste each "CELL" section into a separate Colab cell and run them
in order.  At the end, download all 4 .pt files + classifier.pkl.

Total training time: ~60–90 minutes on free T4 GPU.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
"""
!pip install ultralytics roboflow scikit-learn nltk pandas -q
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Import libraries & verify GPU
# ═══════════════════════════════════════════════════════════════════════════════
"""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")

from ultralytics import YOLO
print("Ultralytics imported successfully ✓")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Authenticate Roboflow (FREE account)
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Go to https://app.roboflow.com → Sign up FREE
# 2. Go to Settings → API Keys → Copy your key
# 3. Paste it below
"""
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_FREE_API_KEY_HERE")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Train MODEL 1 — POTHOLE Detection (YOLOv12)
# ═══════════════════════════════════════════════════════════════════════════════
# Dataset: "pothole-jujbl" on Roboflow Universe (FREE, public)
# URL: https://universe.roboflow.com/atharva-thite/pothole-jujbl
# ~1,100 images, well-annotated for pothole detection
"""
# Download dataset
project  = rf.workspace().project("pothole-jujbl")
dataset1 = project.version(1).download("yolov8")

# Train YOLOv12
model1 = YOLO("yolov12s.pt")   # auto-downloads YOLOv12 small weights
model1.train(
    data    = f"{dataset1.location}/data.yaml",
    epochs  = 30,
    imgsz   = 640,
    batch   = 16,
    name    = "pothole_run",
    project = "civictrack_models",
    patience = 10,        # early stopping
    save    = True,
    verbose = True
)

# Evaluate
metrics1 = model1.val()
print(f"Pothole mAP50: {metrics1.box.map50:.3f}")
print(f"Pothole mAP50-95: {metrics1.box.map:.3f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Train MODEL 2 — GARBAGE Detection (YOLOv12)
# ═══════════════════════════════════════════════════════════════════════════════
# Option A: Roboflow Universe — search "garbage detection"
#   Recommended: "garbage-classification-3" (~2,500 images, free)
#   URL: https://universe.roboflow.com (search "garbage detection")
#
# Option B: Kaggle — "Garbage Detection 6 Categories" (10,464 images)
#   URL: https://www.kaggle.com/datasets
#   Download → Upload to Colab → update path below
"""
# OPTION A: From Roboflow (replace with your found project)
project  = rf.workspace().project("garbage-classification-3")
dataset2 = project.version(1).download("yolov8")

# OPTION B: If using Kaggle dataset, upload and set path:
# dataset2_location = "/content/garbage-detection"

model2 = YOLO("yolov12s.pt")
model2.train(
    data    = f"{dataset2.location}/data.yaml",
    epochs  = 30,
    imgsz   = 640,
    batch   = 16,
    name    = "garbage_run",
    project = "civictrack_models",
    patience = 10,
    save    = True,
    verbose = True
)

metrics2 = model2.val()
print(f"Garbage mAP50: {metrics2.box.map50:.3f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Train MODEL 3 — WATERLOGGING / FLOOD Detection (YOLOv12)
# ═══════════════════════════════════════════════════════════════════════════════
# Dataset: Search "water logging" or "flood detection" on Roboflow Universe
# Recommended: "water-logging" dataset or "flood-detection" dataset
# Both are free and public
"""
# Replace with the exact project name you find on Roboflow Universe
project  = rf.workspace().project("water-logging-aqhov")
dataset3 = project.version(1).download("yolov8")

model3 = YOLO("yolov12s.pt")
model3.train(
    data    = f"{dataset3.location}/data.yaml",
    epochs  = 30,
    imgsz   = 640,
    batch   = 16,
    name    = "waterlog_run",
    project = "civictrack_models",
    patience = 10,
    save    = True,
    verbose = True
)

metrics3 = model3.val()
print(f"Waterlog mAP50: {metrics3.box.map50:.3f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Train MODEL 4 — ROAD CRACK Detection (YOLOv12)
# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Option A: Roboflow Universe — "road crack detection"
#   Many free crack detection datasets available
#
# Dataset Option B: Kaggle — "RDD2022" (47,420 images, multi-national)
#   URL: https://www.kaggle.com/datasets — search "RDD2022 YOLO"
#   This is the most comprehensive road damage dataset available
"""
# OPTION A: From Roboflow
project  = rf.workspace().project("crack-detection-yzwjm")
dataset4 = project.version(1).download("yolov8")

model4 = YOLO("yolov12s.pt")
model4.train(
    data    = f"{dataset4.location}/data.yaml",
    epochs  = 30,
    imgsz   = 640,
    batch   = 16,
    name    = "crack_run",
    project = "civictrack_models",
    patience = 10,
    save    = True,
    verbose = True
)

metrics4 = model4.val()
print(f"Crack mAP50: {metrics4.box.map50:.3f}")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Train NLP text classifier
# ═══════════════════════════════════════════════════════════════════════════════
"""
# Upload train_nlp.py to Colab first, then:
!python train_nlp.py
# This creates: models/classifier.pkl
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Download all trained models
# ═══════════════════════════════════════════════════════════════════════════════
"""
import shutil, os

os.makedirs("final_models", exist_ok=True)

# Copy best weights from each training run
model_files = {
    "pothole.pt":  "civictrack_models/pothole_run/weights/best.pt",
    "garbage.pt":  "civictrack_models/garbage_run/weights/best.pt",
    "waterlog.pt": "civictrack_models/waterlog_run/weights/best.pt",
    "crack.pt":    "civictrack_models/crack_run/weights/best.pt",
}

for dest, src in model_files.items():
    if os.path.exists(src):
        shutil.copy(src, f"final_models/{dest}")
        size_mb = os.path.getsize(f"final_models/{dest}") / (1024*1024)
        print(f"✓ {dest} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {dest} — source not found: {src}")

# Also copy NLP model
if os.path.exists("models/classifier.pkl"):
    shutil.copy("models/classifier.pkl", "final_models/classifier.pkl")
    print("✓ classifier.pkl")

print("\\n── Download final_models/ folder from Colab file browser ──")
print("Place all files in your project's models/ directory")

# Auto-zip for easy download
shutil.make_archive("civictrack_models_final", "zip", "final_models")
print("\\n📦 Download: civictrack_models_final.zip")

from google.colab import files
files.download("civictrack_models_final.zip")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# AFTER TRAINING — Place files in your project:
# ═══════════════════════════════════════════════════════════════════════════════
"""
Your project/models/ folder should contain:
  models/
    pothole.pt       (~20-30 MB)
    garbage.pt       (~20-30 MB)
    waterlog.pt      (~20-30 MB)
    crack.pt         (~20-30 MB)
    classifier.pkl   (~1 MB)

Then test locally:
  pip install -r requirements.txt
  python app.py

Or deploy to Render (see DEPLOYMENT.md).

TIP: If any .pt file > 100 MB, use Git LFS:
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
"""

print("See comments above for step-by-step Colab training instructions.")
print("Each CELL section should be pasted into a separate Colab cell.")
