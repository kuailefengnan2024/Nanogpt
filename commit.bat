@echo off
:: 提交脚本 - 简化git提交过程
:: 用法: commit.bat "提交信息"

:: 检查是否提供了提交信息
if "%~1"=="" (
  echo 错误: 请提供提交信息
  echo 用法: commit.bat "提交信息"
  exit /b 1
)

:: 添加所有更改的文件
echo 正在添加更改的文件...
git add .

:: 提交更改
echo 正在提交更改: %~1
git commit -m "%~1"

:: 推送到远程仓库
echo 正在推送到远程仓库...
git push origin master

echo 提交完成! 