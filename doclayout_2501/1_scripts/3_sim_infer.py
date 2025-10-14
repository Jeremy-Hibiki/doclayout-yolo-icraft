from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from doclayout_yolo.data.augment import LetterBox
from doclayout_yolo.nn.modules.block import DFL
from doclayout_yolo.utils import ops
from doclayout_yolo.utils.tal import dist2bbox, make_anchors
from icraft.host_backend import HostBackend
from icraft.xir import Layout, Network
from icraft.xrt import HostDevice, Session, Tensor
from visualize import DOCLAYOUT_CLASSES, vis

# ---------------------------------参数设置---------------------------------
# 路径设置
GENERATED_JSON_FILE = "../3_deploy/modelzoo/doclayout_yolo/imodel/16/doclayout_yolo_quantized.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/doclayout_yolo/imodel/16/doclayout_yolo_quantized.raw"
IMG_PATH = r"..\3_deploy\modelzoo\doclayout_yolo\io\pdf_imgs\kimi_k2_1.png"

conf_thres = 0.25
max_det = 300
reg_max = 1
nc = len(DOCLAYOUT_CLASSES)


def pred_one_image(img_path):
    # 加载测试图像并转成icraft.Tensor
    img_raw = cv2.imread(img_path)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    im = LetterBox((1280, 960), stride=32, auto=False)(image=img_raw)
    # Img to xir.Tensor
    img_ = np.expand_dims(im, axis=0)
    print("img_ =", img_.shape)
    input_tensor = Tensor(img_, Layout("NHWC"))
    # 加载指令生成后的网络
    generated_network = Network.CreateFromJsonFile(GENERATED_JSON_FILE)
    generated_network.loadParamsFromFile(GENERATED_RAW_FILE)
    print("INFO: Create network!")
    # 创建Session
    session = Session.Create([HostBackend], generated_network.view(0), [HostDevice.Default()])
    session.apply()
    # 模型前向推理
    generated_output = session.forward([input_tensor])
    # 6 out  in n,h,w,c format
    # i = 0  out = (1, 160, 160, 10)
    # i = 1  out = (1, 160, 160, 4)
    # i = 2  out = (1, 80, 80, 10)
    # i = 3  out = (1, 80, 80, 4)
    # i = 4  out = (1, 40, 40, 10)
    # i = 5  out = (1, 40, 40, 4)
    # check outputs
    # for i in range(6):
    #     out = np.array(generated_output[i])
    #     print(out.shape)
    print("INFO: get forward results!")
    # 组装成检测结果
    output_tensors = [torch.from_numpy(np.array(obj)).permute(0, 3, 1, 2).contiguous() for obj in generated_output]

    outputs_n1 = torch.cat((output_tensors[1], output_tensors[0]), 1)  # [1, 14, 160, 160]
    outputs_n2 = torch.cat((output_tensors[3], output_tensors[2]), 1)  # [1, 14, 80, 80]
    outputs_n3 = torch.cat((output_tensors[5], output_tensors[4]), 1)  # [1, 14, 40, 40]
    outputs = [outputs_n1, outputs_n2, outputs_n3]
    print("*" * 80)
    # postprocess - dfl+sigmod
    shape = outputs[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], nc + 4, -1) for xi in outputs], 2)
    box, cls = x_cat.split((reg_max * 4, nc), 1)  # box = [1,4,8400], cls = [1,10,8400]
    dfl_layer = DFL(reg_max) if reg_max > 1 else nn.Identity()
    anchors, strides = (
        x.transpose(0, 1)
        for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32], dtype=np.float32)), 0.5)
    )
    dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)  # [1,14,8400]
    # print(y)
    # print('y = ',y.shape)
    # yolov10 postprocess - NMS free
    preds = y.transpose(-1, -2)

    bboxes, scores, labels = ops.v10postprocess(
        preds, max_det, preds.shape[-1] - 4
    )  # bbox - [1,max_det,4] scores - [1,max_det] labels - [1,300]

    bboxes = ops.xywh2xyxy(bboxes)

    preds = torch.cat(
        [bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1
    )  # [1,max_det,6] = [1,max_det, bbox+scores+label]
    mask = preds[..., 4] > conf_thres
    b, _, c = preds.shape
    preds = preds.view(-1, preds.shape[-1])[mask.view(-1)]  # 取mask = True的结果，即score>conf的结果
    pred = preds.view(b, -1, c)  # [1,res_num,6]
    _, res_num, _ = pred.shape
    # rescale coords to img_raw size
    pred[0, :, :4] = ops.scale_boxes(im.shape[0:2], pred[0, :, :4], img_raw.shape)
    for res in pred[0]:
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
        boxes=pred[0][:, :4],
        scores=pred[0][:, 4],
        cls_ids=pred[0][:, 5],
        conf=conf_thres,
        class_names=DOCLAYOUT_CLASSES,
    )
    cv2.imshow(" ", result_image)
    cv2.waitKey(0)
    print("Detect ", res_num, " objects!")


if __name__ == "__main__":
    if (input_path := Path(IMG_PATH)).is_file():
        pred_one_image(IMG_PATH)
    else:
        for img_path in Path(IMG_PATH).iterdir():
            pred_one_image(str(img_path))
