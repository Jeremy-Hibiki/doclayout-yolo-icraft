# 将模型导出为torch_script
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from doclayout_yolo import YOLOv10
from doclayout_yolo.engine.exporter import Exporter, try_export
from doclayout_yolo.nn.modules.head import v10Detect
from doclayout_yolo.nn.tasks import BaseModel
from doclayout_yolo.utils import LOGGER, colorstr

TRACE_PATH: str = ""


def new_predict_once(self: BaseModel, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        # for export
        if m.i == 23:  # 为了在cat前输出
            return m(x)
        else:
            x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
    return x


# BaseModel._predict_once = new_predict_once


def new_Detect_forward(self: v10Detect, x):
    y = []
    for i in range(self.nl):
        y.append(self.one2one_cv2[i](x[i]))
        y.append(self.one2one_cv3[i](x[i]))
    return y


v10Detect.forward = new_Detect_forward


@try_export
def new_export_torchscript(self, prefix=colorstr("TorchScript:")):
    """YOLOv10 TorchScript model export."""
    global TRACE_PATH

    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    trace_path = TRACE_PATH  # traced path
    im = torch.zeros(1, 3, *self.imgsz, dtype=torch.float32)  # dummy input size
    ts = torch.jit.trace(self.model, im, strict=False)
    extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
    if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        LOGGER.info(f"{prefix} optimizing for mobile...")
        from torch.utils.mobile_optimizer import optimize_for_mobile

        optimize_for_mobile(ts)._save_for_lite_interpreter(trace_path, _extra_files=extra_files)
    else:
        ts.save(trace_path, _extra_files=extra_files)
    return trace_path, None


Exporter.export_torchscript = new_export_torchscript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv10 PyTorch Inference.", add_help=True)
    parser.add_argument(
        "--weights",
        type=str,
        default=(Path(__file__).parent / "../doclayout_yolo_docstructbench_imgsz1280_2501.pt").resolve().as_posix(),
        help="model path(s) for inference.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=(Path(__file__).parent / "../2_compile/fmodel/doclayout_yolo_docstructbench_imgsz1280_2501.pt")
        .resolve()
        .as_posix(),
        help="model path(s) for inference.",
    )
    args = parser.parse_args()
    weights = args.weights  # 权重
    TRACE_PATH = args.save_path  # traced path
    # load model
    model = YOLOv10(weights)

    # export traced model
    success = model.export(nms=True)
    print("Model save at:", success)
