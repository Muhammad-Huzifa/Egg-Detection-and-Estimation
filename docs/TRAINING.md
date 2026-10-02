# Training and export

The existing browser checkpoint is retained at `web-app/model/best.onnx`. Training a new model requires the external dataset and compatible segmentation labels.

Set `ROBOFLOW_API_KEY` in your terminal, then run `python scripts/download_dataset.py`. Download authorization and dataset access depend on your Roboflow account.

If the downloaded labels are bounding boxes, install `requirements-preprocessing.txt`, obtain the SAM ViT-B checkpoint from [the official Segment Anything repository](https://github.com/facebookresearch/segment-anything), and create a separate converted dataset:

```bash
python scripts/preprocessing.py path/to/detection_dataset --output data/segmentation --checkpoint path/to/sam_vit_b_01ec64.pth
```

The output directory must be new and outside the input dataset. Original labels are preserved. SAM pseudo-labels need inspection before training; images without usable polygons may require manual label correction. Check the generated dataset YAML so its paths refer to the converted images and labels, especially if the source YAML uses absolute paths.

```bash
python scripts/train.py data/segmentation --epochs 100
```

The trainer exports its actual best checkpoint and prints the ONNX path. A new checkpoint must satisfy the browser's input/output assumptions before replacing the bundled model. Real training, SAM conversion, export, and browser inference have not been executed in this restructuring.
