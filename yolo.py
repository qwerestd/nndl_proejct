from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n-seg.pt")
    # 导出为 ONNX，开启动态尺寸或固定移动端尺寸（如 384x384 / 480x480）
    model.export(
        format="onnx",
        imgsz=480,  # 与当前输入尺寸一致 (480x480)
        simplify=True,  # 利用 onnxslim 优化算子结构，加快 Android 端加载与执行速度
        opset=12  # 移动端兼容性极佳的算子集版本
    )