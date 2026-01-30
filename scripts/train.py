import torch
from ultralytics import YOLO

def train(dataset_path, epochs=100):
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=20,
        project='runs',
        name='egg_seg',
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    model = YOLO('runs/egg_seg/weights/best.pt')
    model.val()
    model.export(format='onnx', imgsz=640, simplify=True)
    print("Model exported: runs/egg_seg/weights/best.onnx")

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "Eggs-dpy01-1"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    train(dataset_path, epochs)