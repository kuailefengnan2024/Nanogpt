import sys
import os
import torch
import platform
import subprocess

def get_python_info():
    python_version = sys.version.split('\n')[0]
    python_path = sys.executable
    return python_version, python_path

def get_virtual_env(project_path=None):
    # 如果提供了项目路径，检查常见的虚拟环境目录
    if project_path:
        # 检查 venv 文件夹
        venv_path = os.path.join(project_path, "venv")
        if os.path.exists(venv_path):
            return "venv (project local)"
        # 检查 .venv 文件夹（Poetry 等工具常用）
        venv_path = os.path.join(project_path, ".venv")
        if os.path.exists(venv_path):
            return ".venv (project local)"
    # 否则检查当前运行时的环境变量
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        return os.path.basename(venv)
    conda_env = os.environ.get('CONDA_PREFIX')
    if conda_env:
        return os.path.basename(conda_env)
    return "No virtual environment detected"

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

def print_system_info(project_path=None):
    print("=== System Information ===")
    
    python_version, python_path = get_python_info()
    print(f"Python Version: {python_version}")
    print(f"Python Path: {python_path}")
    
    venv = get_virtual_env(project_path)
    print(f"Active Virtual Environment: {venv}")
    
    pytorch_version, cuda_available, cuda_version, device_name = get_pytorch_info()
    print(f"PyTorch Version: {pytorch_version}")
    print(f"CUDA Available: {cuda_available}")
    print(f"CUDA Version: {cuda_version}")
    print(f"GPU Device: {device_name}")
    
    print("======================")

if __name__ == "__main__":
    try:
        # 指定你的项目路径
        PROJECT_PATH = "/path/to/your/project"  # 替换为你的项目路径
        print_system_info(PROJECT_PATH)
    except Exception as e:
        print(f"An error occurred: {str(e)}")