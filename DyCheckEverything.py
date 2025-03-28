import sys
import os
import torch
import platform
import subprocess

def get_python_info():
    # 获取当前Python版本和路径
    python_version = sys.version.split('\n')[0]
    python_path = sys.executable
    return python_version, python_path

def get_virtual_env():
    # 获取当前激活的虚拟环境
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        return os.path.basename(venv)
    return "No virtual environment activated"

def get_pytorch_info():
    # 获取PyTorch版本和CUDA可用性
    try:
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
        else:
            cuda_version = "Not available"
            device_name = "No GPU detected"
        return pytorch_version, cuda_available, cuda_version, device_name
    except ImportError:
        return "PyTorch not installed", False, "N/A", "N/A"

def print_system_info():
    # 获取并打印所有信息
    print("=== System Information ===")
    
    # Python信息
    python_version, python_path = get_python_info()
    print(f"Python Version: {python_version}")
    print(f"Python Path: {python_path}")
    
    # 虚拟环境信息
    venv = get_virtual_env()
    print(f"Active Virtual Environment: {venv}")
    
    # PyTorch信息
    pytorch_version, cuda_available, cuda_version, device_name = get_pytorch_info()
    print(f"PyTorch Version: {pytorch_version}")
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Version: {cuda_version}")
    print(f"GPU Device: {device_name}")
    
    print("======================")

if __name__ == "__main__":
    try:
        print_system_info()
    except Exception as e:
        print(f"An error occurred: {str(e)}")