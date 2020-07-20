# TensorRT_yolo3_module

------

## 1. Install TensorRT on Ubuntu
cuda10.2  
tensorrt7.1.0  
pycuda2019.1.1  
onnx1.5.0
## 2. Test TensorRT_yolo3_module
- `cd convert `
- `python3 yolov3_to_onnx.py`
- `python3 onnx_to_trt_1batch.py`

## 3. Import TensorRT_yolo3_module
- This project has been packaged into **class**, so you can use it directly according `import xx` command.
