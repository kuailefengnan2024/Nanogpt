# install_pytorch.py
import subprocess
import sys

def run_command(command):
    process = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)
    return process.returncode == 0

def install_pytorch_with_cuda():
    # 检查当前环境
    print("当前 Python 路径:", sys.executable)
    
    # 安装 PyTorch with CUDA 11.8
    conda_cmd = (
        "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 "
        "-c pytorch -c nvidia -y"
    )
    print("正在安装 PyTorch with CUDA 11.8...")
    success = run_command(conda_cmd)
    
    if success:
        print("安装完成，请重新运行脚本检查 PyTorch 状态。")
    else:
        print("安装失败，请检查错误信息。")

if __name__ == "__main__":
    install_pytorch_with_cuda()