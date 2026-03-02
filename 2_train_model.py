#!/usr/bin/env python3
from pathlib import Path
import os

def main():
    # --- Paths ---
    root = Path(__file__).resolve().parent
    data_yaml = root / "rock-paper-scissors-1" / "data.yaml"
    runs_dir = root / "train_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find dataset config: {data_yaml}")

    # Import after basic checks
    from ultralytics import YOLO

    # --- CPU-only safe defaults for Codespaces (8GB RAM, no swap) ---
    # You can override from terminal with env vars if you want:
    #   YOLO_EPOCHS=20 YOLO_IMGSZ=416 YOLO_BATCH=8 python 2_train_model.py
    epochs  = int(os.getenv("YOLO_EPOCHS", "10"))
    imgsz   = int(os.getenv("YOLO_IMGSZ", "640"))   # 640 is heavier on CPU/RAM
    batch   = int(os.getenv("YOLO_BATCH", "8"))     # keep small for 8GB RAM
    workers = int(os.getenv("YOLO_WORKERS", "0"))   # 0 is safest in Codespaces

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",          # <-- CPU ONLY
        workers=workers,
        amp=False,             # <-- disable AMP on CPU
        cache=False,           # <-- avoid memory spikes
        project=str(runs_dir),
        name=os.getenv("YOLO_RUN_NAME", "yolov26n_cpu_codespaces"),
        exist_ok=True
    )

    print("\n✅ Done training! Check:", runs_dir)

if __name__ == "__main__":
    main()