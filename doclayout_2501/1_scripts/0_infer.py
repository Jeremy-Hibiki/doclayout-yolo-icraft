# 通过 opencv 读取了一张图像，并送入模型中推理得到输出 results，
# results 中保存着不同任务的结果，我们这里是检测任务，因此只需要拿到对应的 boxes 即可
import argparse

import cv2
from doclayout_yolo import YOLOv10


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv10 PyTorch Inference.", add_help=True)
    parser.add_argument(
        "--weights",
        type=str,
        help="model path(s) for inference.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="the source path, e.g. image-file.",
    )
    parser.add_argument("--res", type=str, default="predict.png", help="the res path.")
    args = parser.parse_args()
    weights = args.weights  # 权重
    img_path = args.source  # 推理图片
    res_path = args.res  # 结果存放路径
    # load model
    model = YOLOv10(weights)
    # read img
    img = cv2.imread(img_path)
    assert img is not None, "Image Not Found " + img_path
    # get pred res
    det_res = model.predict(source=img)
    annotated_frame = det_res[0].plot(pil=True, line_width=2, font_size=24)
    # save res
    cv2.imwrite(res_path, annotated_frame)
    print("save at ", res_path)
