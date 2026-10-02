import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download the original Roboflow egg dataset using your configured API key.")
    parser.add_argument("--workspace", default="digital-image-proecessing")
    parser.add_argument("--project", default="eggs-dpy01-yqgdf")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        parser.error("Set ROBOFLOW_API_KEY before downloading the dataset.")
    from roboflow import Roboflow
    dataset = Roboflow(api_key=key).workspace(args.workspace).project(args.project).version(args.version).download("yolov8")
    print("Dataset:", dataset.location)

if __name__ == "__main__":
    main()
