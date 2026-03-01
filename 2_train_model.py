#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path

def main():
    # Make sure we run relative to this script's folder
    root = Path(__file__).resolve().parent
    data_yaml = root / "rock-paper-scissors-1" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find {data_yaml}")

    # Load pretrained YOLOv26n weights
    model = YOLO("yolo26n.pt")

    # Train
    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=640,
        batch=-1,     # auto batch size (great for RTX 4070 Super)
        device=0,     # use GPU
        workers=8,
        cache=True,   # speeds up training
        project=str(root / "train_runs"),
        name="yolov26n_rps"
    )

    print("\n✅ Done training!")
if __name__ == "__main__":
    main()
