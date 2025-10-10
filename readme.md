# Doclayout YOLO

- **Model**: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501
- **Model Architecture**: Yolo10, 1280x1280, ==without DFL==
- **Model Task**: document layout detection, into 10 classes
- **Use Case**: check https://github.com/opendatalab/DocLayout-YOLO

icraft version: v3.7.1

---

## Infer with original model

```shell
cd ./1_scripts
python ./0_infer.py --weights ../doclayout_yolo_docstructbench_imgsz1280_2501.pt --source ./imgs/page_4.png
```

## Trace and infer

```shell
cd ./1_scripts
python ./1_save.py --weights ../doclayout_yolo_docstructbench_imgsz1280_2501.pt --save_path ../2_compile/fmodel/doclayout_yolo_docstructbench_imgsz1280_2501.pt
python ./2_save_infer.py --model ../2_compile/fmodel/doclayout_yolo_docstructbench_imgsz1280_2501.pt --source ../2_compile/qtset/pdf_imgs/page_4.png --imgsz 1280
```

## Compile and simulate

```shell
cd ../2_compile
icraft compile ./config/yolov10n_8.toml

cd ../1_scripts
python ./3_sim_infer.py
```
