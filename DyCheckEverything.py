import sys
import os
import torch
import platform
import subprocess

def get_python_info():
    python_version = sys.version.split('\n')[0]
    python_path = sys.executable
    return python_version, python_path

def get_virtual_env():
    # 检查传统虚拟环境的 VIRTUAL_ENV
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        return os.path.basename(venv)
    # 检查 Anaconda 环境的 CONDA_PREFIX
    conda_env = os.environ.get('CONDA_PREFIX')
    if conda_env:
        return os.path.basename(conda_env)
    return "No virtual environment activated"

def get_pytorch_info():
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
    print("=== System Information ===")
    
    python_version, python_path = get_python_info()
    print(f"Python Version: {python_version}")
    print(f"Python Path: {python_path}")
    
    venv = get_virtual_env()
    print(f"Active Virtual Environment: {venv}")
    
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