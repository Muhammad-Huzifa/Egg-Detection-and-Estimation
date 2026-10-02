# Browser implementation

Serve `web-app/` over HTTP locally, or HTTPS when using the camera remotely. Opening the HTML directly through file:// is not the supported model-loading workflow.

The page loads ONNX Runtime Web from a CDN and the bundled `model/best.onnx`. Internet access is required for the runtime script. The code expects a 640×640 RGB tensor named `images` and a compatible `output0` detection tensor. The provided binary has been preserved; its actual tensor contract has not been independently inspected here.

Camera mode, captured frames, and uploaded images use the same detector. Bounding-box aspect ratios are mapped heuristically to displayed size categories and approximate millimeters. There is no camera calibration or reference object; these are not physical measurements. Pixel box areas are not segmentation-mask areas.
