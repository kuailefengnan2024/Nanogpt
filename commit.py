#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提交脚本 - 简化git提交过程
用法: python commit.py "提交信息"
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

def commit(message):
    """添加、提交并推送更改"""
    # 添加所有更改的文件
    print("正在添加更改的文件...")
    if not run_command("git add ."):
        return False
    
    # 提交更改
    print(f"正在提交更改: {message}")
    if not run_command(f'git commit -m "{message}"'):
        return False
    
    # 推送到远程仓库
    print("正在推送到远程仓库...")
    if not run_command("git push origin master"):
        return False
    
    print("提交完成!")
    return True

if __name__ == "__main__":
    # 检查是否提供了提交信息
    if len(sys.argv) < 2:
        print("错误: 请提供提交信息")
        print('用法: python commit.py "提交信息"')
        sys.exit(1)
    
    # 获取提交信息
    commit_message = sys.argv[1]
    
    # 执行提交
    if not commit(commit_message):
        sys.exit(1) 