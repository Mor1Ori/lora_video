import torch
import cv2
import diffusers

# 1. 检查 GPU 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. 检查核心库版本
print(f"Diffusers Version: {diffusers.__version__}")
print(f"OpenCV Version: {cv2.__version__}")

# 如果以上没有报错，且 CUDA Available 输出为 True，则环境搭建大功告成！