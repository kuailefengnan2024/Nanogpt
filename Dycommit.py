#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提交脚本 - 简化git提交过程
用法: python commit.py ["提交信息"]
如果不提供提交信息，将使用当前时间作为提交信息
"""

import os
import sys
import subprocess
import datetime

def run_command(command):
    """执行命令并返回结果"""
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        shell=True, 
        universal_newlines=True,
        encoding='utf-8'  # 明确指定编码为UTF-8
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"错误: {stderr.strip()}")  # 使用strip()去除多余空白
        return False
    
    if stdout:
        print(stdout.strip())  # 只在有输出时打印
    return True

def commit(message):
    """添加、提交并推送更改"""
    # 检查Git配置
    print("检查Git配置...")
    if not run_command("git config user.name") or not run_command("git config user.email"):
        print("请先配置Git用户名和邮箱:")
        print("git config --global user.name \"你的名字\"")
        print("git config --global user.email \"你的邮箱\"")
        return False
    
    # 检查是否有更改需要提交
    print("检查是否有更改...")
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True, encoding='utf-8')
    if not result.stdout.strip():
        print("没有需要提交的更改")
        return False
    
    # 添加所有更改的文件
    print("正在添加更改的文件...")
    if not run_command("git add ."):
        return False
    
    # 提交更改
    print(f"正在提交更改: {message}")
    if not run_command(f'git commit -m "{message}"'):
        return False
    
    # 获取当前分支
    branch_result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True, encoding='utf-8')
    current_branch = branch_result.stdout.strip() or "master"  # 如果为空则默认为master
    
    # 推送到远程仓库
    print(f"正在推送到远程仓库 (分支: {current_branch})...")
    if not run_command(f"git push origin {current_branch}"):
        return False
    
    print("提交完成!")
    return True

if __name__ == "__main__":
    # 检查是否提供了提交信息
    if len(sys.argv) < 2:
        # 使用当前时间作为默认提交信息
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"自动提交于 {current_time}"
        print(f"未提供提交信息，使用默认时间信息: {commit_message}")
    else:
        # 获取提交信息
        commit_message = sys.argv[1]
    
    # 执行提交
    if not commit(commit_message):
        sys.exit(1) 