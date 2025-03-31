#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拉取脚本 - 简化git拉取更新过程
用法: python pull.py [远程仓库名称] [分支名称]
如果不提供参数，将从origin仓库的master分支拉取
"""

import os
import sys
import subprocess

def run_command(command):
    """执行命令并返回结果"""
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        shell=True, 
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"错误: {stderr}")
        return False
    
    print(stdout)
    return True

def pull(remote="origin", branch="master"):
    """从远程仓库拉取更新"""
    print(f"正在从 {remote}/{branch} 拉取更新...")
    if not run_command(f"git pull {remote} {branch}"):
        return False
    
    print("拉取完成!")
    return True

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        remote = "origin"
    else:
        remote = sys.argv[1]
        
    if len(sys.argv) < 3:
        branch = "master"
    else:
        branch = sys.argv[2]
        
    print(f"将从 {remote}/{branch} 拉取更新")
    
    # 执行拉取
    if not pull(remote, branch):
        sys.exit(1) 