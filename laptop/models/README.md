# Models

ONNX files live here and are **gitignored** (`laptop/models/*.onnx`). Fetch them with:

```bash
python tools/download_model.py          # from laptop/; idempotent, verifies the sha256
```

## Object detector: RT-DETR v2, ResNet-18 backbone, COCO

| | |
| --- | --- |
| File | `models/rtdetr_v2_r18vd.onnx` (fp32) |
| Source | https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX, file `onnx/model.onnx` |
| Pinned revision | `936f90b6a476c6da4dfe053fc521af55285976ba` |
| URL | `https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/resolve/936f90b6a476c6da4dfe053fc521af55285976ba/onnx/model.onnx` |
| sha256 | `583a236ac21c95a7fd94f284fc21485e42355bfef82c27011ba78fbc09ee87e2` |
| Size | 81,057,510 bytes |
| Input | `pixel_values` float32 `[1, 3, 640, 640]`, RGB, /255 only (no mean/std), plain stretch resize |
| Output | `logits [1, 300, 80]` (sigmoid scores), `pred_boxes [1, 300, 4]` normalized cx, cy, w, h |
| Classes | COCO 80; PointView keeps person, bicycle, car, motorcycle, bus, truck |

### License evidence (checked 2026-10-26)

* ONNX export, model card metadata `license: apache-2.0`: https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX
* Original weights, `license: apache-2.0`: https://huggingface.co/PekingU/rtdetr_v2_r18vd
* Original RT-DETR code, Apache License 2.0: https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE
* Inference code is our own (ONNX Runtime, MIT) and does not use Ultralytics or any AGPL code.
* Training data is COCO (annotations CC BY 4.0; images from Flickr under various licenses). The weights
  are not a copy of the images; keep the COCO citation in mind if you redistribute the model.

### Why this model

Apache-2.0 (design doc requirement), already exported to ONNX with dynamic shapes, NMS-free (fewer
post-processing steps to get wrong), and small enough for a laptop GPU/NPU via DirectML. D-FINE
(`onnx-community/dfine_s_coco-ONNX`, 41 MB) is a possible faster/lighter alternative; its card has no
license field set in the Hub metadata, so it was not chosen until that is clarified.

### Measured CPU speed (no GPU in the dev container)

About 160-185 ms per frame on 4 CPU cores with onnxruntime CPU (preprocess + inference + postprocess).
That is the CPU fallback only; the DirectML number on the real laptop is not measured yet.
