import argparse
from pathlib import Path

def train(dataset_path, epochs=100):
    import torch
    from ultralytics import YOLO
    root = Path(__file__).resolve().parents[1]
    model = YOLO("yolov8n-seg.pt")
    model.train(data=str(Path(dataset_path).resolve() / "data.yaml"), epochs=epochs, imgsz=640, batch=16, patience=20, project=str(root / "runs"), name="egg_seg", device=0 if torch.cuda.is_available() else "cpu")
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    trained.val(data=str(Path(dataset_path).resolve() / "data.yaml"))
    exported = trained.export(format="onnx", imgsz=640, simplify=True)
    print("Best checkpoint:", best)
    print("Exported model:", exported)

def main():
    parser = argparse.ArgumentParser(description="Train and export an egg segmentation model.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if not (args.dataset / "data.yaml").is_file():
        parser.error("Dataset must contain data.yaml and segmentation labels.")
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    train(args.dataset, args.epochs)

if __name__ == "__main__":
    main()
