import cv2
import numpy as np
import torch
import torch.jit
from doclayout_yolo.data.augment import LetterBox
from icraft.host_backend import HostBackend
from icraft.xir import Layout, Network
from icraft.xrt import HostDevice, Session, Tensor
from visualize import DOCLAYOUT_CLASSES

conf_thres = 0.25
max_det = 300
reg_max = 1
nc = len(DOCLAYOUT_CLASSES)
imgsz = (1280, 960)
GENERATED_JSON_FILE = "../3_deploy/modelzoo/doclayout_yolo/imodel/8/doclayout_yolo_parsed.json"
GENERATED_RAW_FILE = "../3_deploy/modelzoo/doclayout_yolo/imodel/8/doclayout_yolo_parsed.raw"
IMG_PATH = "./imgs/page_4.png"
MODEL_PATH = "../2_compile/fmodel/doclayout_yolo_docstructbench_imgsz1280_2501.pt"


def torch_pred(img_path, model_path):
    global conf_thres, max_det, reg_max, nc, imgsz

    img_raw = cv2.imread(img_path)
    # 前处理
    letterbox = LetterBox(imgsz, auto=False, stride=32)
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
    print("*" * 80)
    # 结果重组
    outputs_n1 = torch.cat((output[0], output[1]), 1)  # [1, 14, 160, 160]
    outputs_n2 = torch.cat((output[2], output[3]), 1)  # [1, 14, 80, 80]
    outputs_n3 = torch.cat((output[4], output[5]), 1)  # [1, 14, 40, 40]
    outputs = [outputs_n1, outputs_n2, outputs_n3]

    # x_cat = torch.cat([xi.view(xi.shape[0], nc + 4, -1) for xi in outputs], 2)
    # box, cls = x_cat.split((4, nc), 1)
    return outputs


def icraft_pred(img_path):
    # 加载测试图像并转成icraft.Tensor
    img_raw = cv2.imread(img_path)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    im = LetterBox(imgsz, stride=32, auto=False, center=True)(image=img_raw)
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
    session.setLogIO(True)
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

    outputs_n1 = torch.cat((output_tensors[0], output_tensors[1]), 1)  # [1, 14, 160, 160]
    outputs_n2 = torch.cat((output_tensors[2], output_tensors[3]), 1)  # [1, 14, 80, 80]
    outputs_n3 = torch.cat((output_tensors[4], output_tensors[5]), 1)  # [1, 14, 40, 40]
    outputs = [outputs_n1, outputs_n2, outputs_n3]
    return outputs


if __name__ == "__main__":
    torch_outputs = torch_pred(IMG_PATH, MODEL_PATH)
    icraft_outputs = icraft_pred(IMG_PATH)

    for x1, x2 in zip(torch_outputs, icraft_outputs):
        delta = (x1 - x2).abs()
        print(delta.shape, delta.min(), delta.max(), delta.mean(), delta.std())
