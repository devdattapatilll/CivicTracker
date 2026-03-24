"""
train_yolo_local.py
-------------------
Downloads free public datasets and trains all 4 YOLO models locally.
Uses minimal epochs on CPU — retrain in Colab with more epochs for production.
"""
import os, sys, shutil

def check_roboflow():
    try:
        from roboflow import Roboflow
        return True
    except ImportError:
        return False

def train_model(dataset_path, model_name, epochs=3, imgsz=320, base_model="yolov8n.pt"):
    """Train a single YOLO model."""
    from ultralytics import YOLO
    import math

    # Patch AAttn for YOLOv12 compat
    try:
        import torch
        from ultralytics.nn.modules.block import AAttn
        if not getattr(AAttn, "_patched", False):
            orig = AAttn.forward
            def _fwd(self, x):
                try:
                    return orig(self, x)
                except AttributeError as e:
                    if not (hasattr(self, "qk") and hasattr(self, "v") and "qkv" in str(e)):
                        raise
                    b, c, h, w = x.shape
                    n = h * w
                    if hasattr(self, "pe"):
                        x = x + self.pe(x)
                    qk = self.qk(x).flatten(2).transpose(1, 2)
                    v  = self.v(x).flatten(2).transpose(1, 2)
                    q, k = qk.chunk(2, dim=-1)
                    heads = int(getattr(self, "num_heads", 8))
                    head_dim = c // heads if heads else c
                    scale = 1.0 / math.sqrt(head_dim) if head_dim else 1.0
                    q = q.view(b, n, heads, head_dim).transpose(1, 2)
                    k = k.view(b, n, heads, head_dim).transpose(1, 2)
                    v = v.view(b, n, heads, head_dim).transpose(1, 2)
                    attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
                    out = (attn @ v).transpose(1, 2).contiguous().view(b, n, c)
                    out = out.transpose(1, 2).view(b, c, h, w)
                    if hasattr(self, "proj"):
                        out = self.proj(out)
                    return out
            AAttn.forward = _fwd
            AAttn._patched = True
            print("[Patch] AAttn patched for YOLOv12")
    except Exception:
        pass

    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"  ERROR: {yaml_path} not found!")
        return None

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Epochs:   {epochs} (CPU mode)")
    print(f"  ImgSize:  {imgsz}")
    print(f"{'='*60}")

    try:
        # Use specified base model, fall back to YOLOv8n
        try:
            model = YOLO(base_model)
            print(f"  Using {base_model} base model")
        except Exception:
            fallback = "yolov8n.pt"
            print(f"  {base_model} not available, using {fallback} as fallback")
            model = YOLO(fallback)

        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=4,         # small batch for CPU
            device="cpu",
            workers=0,
            name=f"{model_name}_run",
            project="civictrack_training",
            patience=epochs,  # no early stopping
            save=True,
            verbose=True,
            exist_ok=True,
        )

        # Copy best weights
        best_path = f"civictrack_training/{model_name}_run/weights/best.pt"
        last_path = f"civictrack_training/{model_name}_run/weights/last.pt"
        dest = f"models/{model_name}.pt"

        src = best_path if os.path.exists(best_path) else last_path
        if os.path.exists(src):
            os.makedirs("models", exist_ok=True)
            shutil.copy(src, dest)
            size_mb = os.path.getsize(dest) / (1024*1024)
            print(f"  ✓ Saved: {dest} ({size_mb:.1f} MB)")
            return dest
        else:
            print(f"  ERROR: No weights found at {best_path} or {last_path}")
            return None
    except Exception as e:
        print(f"  ERROR training {model_name}: {e}")
        return None


def download_from_roboflow(project_url, version=1, format="yolov8"):
    """Download a dataset from Roboflow Universe (requires free API key)."""
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("  No ROBOFLOW_API_KEY set. Using alternative dataset sources.")
        return None

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_url)
        dataset = project.version(version).download(format)
        return dataset.location
    except Exception as e:
        print(f"  Roboflow download failed: {e}")
        return None


def create_sample_dataset(name, num_images=20):
    """Create a minimal synthetic dataset for testing when real data isn't available."""
    import numpy as np
    from PIL import Image

    base = f"datasets/{name}"
    for split in ["train", "valid"]:
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        n = num_images if split == "train" else max(4, num_images // 4)
        for i in range(n):
            # Create a simple synthetic image with random colored rectangles
            img = np.random.randint(50, 200, (320, 320, 3), dtype=np.uint8)
            # Add a "defect" region
            x1, y1 = np.random.randint(20, 200, 2)
            w, h = np.random.randint(40, 100, 2)
            color = np.random.randint(0, 50, 3)
            img[y1:y1+h, x1:x1+w] = color

            Image.fromarray(img).save(os.path.join(img_dir, f"{i:04d}.jpg"))

            # YOLO format annotation: class x_center y_center width height (normalized)
            cx = (x1 + w/2) / 320
            cy = (y1 + h/2) / 320
            nw = w / 320
            nh = h / 320
            with open(os.path.join(lbl_dir, f"{i:04d}.txt"), "w") as f:
                f.write(f"0 {cx:.4f} {cy:.4f} {nw:.4f} {nh:.4f}\n")

    # Create data.yaml
    class_name = {
        "pothole": "pothole",
        "garbage": "garbage",
        "waterlog": "waterlogging",
        "crack": "crack"
    }.get(name, name)

    yaml_content = f"""train: {os.path.abspath(os.path.join(base, 'train', 'images'))}
val: {os.path.abspath(os.path.join(base, 'valid', 'images'))}
nc: 1
names: ['{class_name}']
"""
    yaml_path = os.path.join(base, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  Created synthetic dataset: {base} ({num_images} train + {max(4, num_images//4)} val images)")
    return base


def main():
    print("=" * 60)
    print("CivicTrack YOLO Model Training (CPU Mode)")
    print("=" * 60)
    print()
    print("NOTE: Training on CPU with minimal epochs for quick model")
    print("creation. For production accuracy, retrain in Google Colab")
    print("with T4 GPU using TRAIN_IN_COLAB.py!")
    print()

    os.makedirs("models", exist_ok=True)

    models_to_train = ["pothole", "garbage", "waterlog", "crack"]
    roboflow_projects = {
        "pothole": "pothole-jujbl",
        "garbage": "garbage-classification-3",
        "waterlog": "water-logging-aqhov",
        "crack": "crack-detection-yzwjm",
    }

    results = {}

    for name in models_to_train:
        print(f"\n{'─'*60}")
        print(f"  Model: {name}")
        print(f"{'─'*60}")

        dataset_path = None

        # Try Roboflow first
        if check_roboflow() and os.environ.get("ROBOFLOW_API_KEY"):
            print("  Attempting Roboflow download...")
            dataset_path = download_from_roboflow(roboflow_projects[name])

        # Fall back to synthetic dataset
        if not dataset_path:
            print("  Using synthetic dataset for quick model creation...")
            dataset_path = create_sample_dataset(name, num_images=30)

        # Train — YOLOv12s for pothole (user requirement), YOLOv8n for others
        bm = "yolov12s.pt" if name == "pothole" else "yolov8n.pt"
        result = train_model(dataset_path, name, epochs=3, imgsz=320, base_model=bm)
        results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, path in results.items():
        if path:
            size = os.path.getsize(path) / (1024*1024)
            print(f"  ✓ {name:12s} → {path} ({size:.1f} MB)")
        else:
            print(f"  ✗ {name:12s} → FAILED")

    # Check if all models exist
    all_ok = all(os.path.exists(f"models/{n}.pt") for n in models_to_train)
    nlp_ok = os.path.exists("models/classifier.pkl")

    print(f"\n  NLP classifier: {'✓' if nlp_ok else '✗'}")
    print(f"\n  All models ready: {'YES ✓' if all_ok and nlp_ok else 'PARTIAL'}")

    if all_ok:
        print("\n  You can now start the backend:")
        print("    python app.py")
        print("\n  For better accuracy, retrain in Google Colab with:")
        print("    TRAIN_IN_COLAB.py (30 epochs, T4 GPU)")


if __name__ == "__main__":
    main()
