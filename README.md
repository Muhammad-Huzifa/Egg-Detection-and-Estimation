# Egg Detection and Size Estimation

A browser-based egg-detection prototype using ONNX Runtime Web, with supporting Python scripts for dataset download, SAM-assisted label preparation, YOLO segmentation training, and ONNX export.

The interface supports camera frames, photo capture, image upload, and annotated image download. Its size categories are heuristic estimates derived from bounding-box proportions; they are not calibrated measurements.

## Run the browser application

```bash
git clone https://github.com/Muhammad-Huzifa/Egg-Detection-and-Estimation.git
cd Egg-Detection-and-Estimation
python -m http.server 8000 --directory web-app
```

Open http://localhost:8000. The existing `web-app/model/best.onnx` is included. No Python ML installation is needed to serve the static page. Allow camera access when using camera mode; the CDN runtime requires internet access. Read [browser assumptions](docs/BROWSER.md).

## Python training tools

Use Python 3.11 and a separate virtual environment:

```bash
python -m venv .venv
```

| Terminal | Activation |
| --- | --- |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

After activation:

```bash
python -m pip install -r requirements.txt
python scripts/download_dataset.py --help
python scripts/train.py --help
```

The dataset downloader reads `ROBOFLOW_API_KEY`. See [training instructions](docs/TRAINING.md) for label format, optional SAM dependencies, commands, and checkpoint export. Label conversion now writes a new dataset copy rather than changing the original labels.

## Structure

| Path | Purpose |
| --- | --- |
| `web-app/index.html` | Application page |
| `web-app/styles.css` | Page styling |
| `web-app/script.js` | Image preprocessing, inference, overlays, and interaction |
| `web-app/model/best.onnx` | Existing exported model |
| `scripts/` | Dataset, preprocessing, and training commands |
| `docs/` | Browser contract and training guide |

Python and JavaScript syntax and CLI argument handling were checked. Five dataset-copy protection checks passed with SAM replaced by a fixture. Real browser inference, camera access, dataset download, SAM conversion, and retraining have not been verified here.

Run the dataset-copy checks after installing NumPy:

```bash
python -m unittest discover -s tests -v
```

## Team

- [Muhammad Huzifa](https://www.linkedin.com/in/muhammad-huzifa3202/)
- [Muhammad Abbas](https://www.linkedin.com/in/muhammad-abbas-b93524279/)
