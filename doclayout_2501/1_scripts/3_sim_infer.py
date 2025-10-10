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
GENERATED_JSON_FILE = "../3_deploy/modelzoo/yolov10/imodel/8/doclayout_yolo_quantized.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/yolov10/imodel/8/doclayout_yolo_quantized.raw"
IMG_PATH = "./imgs/page_4.png"

conf_thres = 0.25
max_det = 300
reg_max = 1
nc = len(DOCLAYOUT_CLASSES)

# 加载测试图像并转成icraft.Tensor
img_raw = cv2.imread(IMG_PATH)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
im = LetterBox((1280, 1280), stride=32, auto=False)(image=img_raw)
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
# i = 0  out = (1, 80, 80, 80)
# i = 1  out = (1, 80, 80, 64)
# i = 2  out = (1, 40, 40, 80)
# i = 3  out = (1, 40, 40, 64)
# i = 4  out = (1, 20, 20, 80)
# i = 5  out = (1, 20, 20, 64)
# check outputs
# for i in range(6):
#     out = np.array(generated_output[i])
#     print(out.shape)
print("INFO: get forward results!")
# 组装成检测结果
output_tensors = [torch.from_numpy(np.array(obj)).permute(0, 3, 1, 2).contiguous() for obj in generated_output]
for out in output_tensors:
    print(out.shape, out.min(), out.max(), out.mean())

outputs_n1 = torch.cat((output_tensors[0], output_tensors[1]), 1)  # [1, 14, 160, 160]
outputs_n2 = torch.cat((output_tensors[2], output_tensors[3]), 1)  # [1, 14, 80, 80]
outputs_n3 = torch.cat((output_tensors[4], output_tensors[5]), 1)  # [1, 14, 40, 40]
outputs = [outputs_n1, outputs_n2, outputs_n3]
print("*" * 80)
# postprocess - dfl+sigmod
shape = outputs[0].shape  # BCHW
x_cat = torch.cat([xi.view(shape[0], nc + 4, -1) for xi in outputs], 2)
box, cls = x_cat.split((reg_max * 4, nc), 1)  # box = [1,64,8400], cls = [1,80,8400]
dfl_layer = DFL(reg_max) if reg_max > 1 else nn.Identity()
anchors, strides = (
    x.transpose(0, 1) for x in make_anchors(outputs, torch.from_numpy(np.array([8, 16, 32], dtype=np.float32)), 0.5)
)
dbox = dist2bbox(dfl_layer(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
y = torch.cat((dbox, cls.sigmoid()), 1)  # [1,84,8400]
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
