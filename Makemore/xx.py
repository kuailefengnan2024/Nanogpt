import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
print(torch.version.cuda) # 查看 PyTorch 编译时使用的 CUDA 版本