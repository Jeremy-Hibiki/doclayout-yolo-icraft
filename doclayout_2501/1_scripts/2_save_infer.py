import argparse
import os
import pathlib

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn as nn
from doclayout_yolo.data.augment import LetterBox
from doclayout_yolo.nn.modules.block import DFL
from doclayout_yolo.utils import ops
from doclayout_yolo.utils.tal import dist2bbox, make_anchors
from visualize import DOCLAYOUT_CLASSES, vis

conf_thres = 0.25
max_det = 300
reg_max = 1
nc = len(DOCLAYOUT_CLASSES)


def pred_one_image(img_path, model_path, test_size):
    global conf_thres, max_det, reg_max, nc

    img_raw = cv2.imread(img_path)
    # 前处理
    letterbox = LetterBox(test_size, auto=False, stride=32)
    im = np.stack([letterbox(image=x) for x in [img_raw]])
    print("******im =", im.shape)

    im = im[..., ::-1].transpose((0, 3, 1, 2))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    im = im.float()
    im /= 255
    # 加载traced模型
    model = torch.jit.load(model_path)
    output = model(im)
    for out in output:
        print(out.shape, out.min(), out.max(), out.mean())
    print("*" * 80)
    # 结果重组
    outputs_n1 = torch.cat((output[0], output[1]), 1)  # [1, 14, 160, 160]
    outputs_n2 = torch.cat((output[2], output[3]), 1)  # [1, 14, 80, 80]
    outputs_n3 = torch.cat((output[4], output[5]), 1)  # [1, 14, 40, 40]
    outputs = [outputs_n1, outputs_n2, outputs_n3]

    x_cat = torch.cat([xi.view(xi.shape[0], nc + 4, -1) for xi in outputs], 2)
    box, cls = x_cat.split((4, nc), 1)

    dfl_layer = DFL(reg_max) if reg_max > 1 else nn.Identity()
    anchors, strides = (
        x.transpose(0, 1)
        for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32], dtype=np.float32)), 0.5)
    )

    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides

    preds = torch.cat((dbox, cls.sigmoid()), 1)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds.transpose(-1, -2)
    bboxes, scores, labels = ops.v10postprocess(preds, max_det, nc)
    bboxes = ops.xywh2xyxy(bboxes)
    preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    mask = preds[..., 4] > conf_thres

    preds = [p[mask[idx]] for idx, p in enumerate(preds)]
    pred = preds[0]
    res_num = len(pred)

    # rescale coords to img_raw size
    pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], img_raw.shape)
    for res in pred:
        x0 = int(res[0])
        y0 = int(res[1])
        x1 = int(res[2])
        y1 = int(res[3])
        score = res[4]
        cls = int(res[5])
        label = DOCLAYOUT_CLASSES[cls]
        print(f"{label}: {score * 100:.2f}%  ({x0}, {y0}), ({x1}, {y1})")
    # show results
    result_image = vis(
        img_raw,
        boxes=pred[:, :4],
        scores=pred[:, 4],
        cls_ids=pred[:, 5],
        conf=conf_thres,
        class_names=DOCLAYOUT_CLASSES,
    )
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)
    print("Detect ", res_num, " objects!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=(pathlib.Path(__file__).parent / "../2_compile/fmodel/doclayout_yolo_docstructbench_imgsz1280_2501.pt")
        .resolve()
        .as_posix(),
        help="torchscript model path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=(pathlib.Path(__file__).parent / "../2_compile/qtset/pdf_imgs/page_4.png").resolve().as_posix(),
        help="image path",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[1280, 960], help="image size")
    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    test_size = tuple(opt.imgsz)

    if pathlib.Path(opt.source).is_file():
        pred_one_image(opt.source, opt.model, test_size)
    elif pathlib.Path(opt.source).is_dir():
        image_list = os.listdir(opt.source)
        for image_file in image_list:
            image_path = opt.source + "//" + image_file
            pred_one_image(image_path, opt.model, test_size)
