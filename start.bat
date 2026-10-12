@echo off
chcp 65001 >nul
title nanoStudio - 极简通用多模态深度学习实验台

echo ================================================================
echo               ⚡ nanoStudio 一键启动器 (Windows)
echo      极简通用多模态深度学习实验台 · LLM / Vision / Diffusion
echo ================================================================
echo.

:: 1. 检查 Python 环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到系统 Python 环境！
    echo 请先安装 Python (推荐 3.9+) 并确保勾选了 "Add Python to PATH"。
    echo.
    pause
    exit /b 1
)

:: 2. 如果存在虚拟环境则自动激活
if exist venv\Scripts\activate.bat (
    echo [环境] 正在激活虚拟环境 (venv)...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo [环境] 正在激活虚拟环境 (.venv)...
    call .venv\Scripts\activate.bat
)

:: 3. 校验核心依赖
echo [环境] 正在校验 PyTorch 与 Pillow 依赖...
python -c "import torch, PIL, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 检测到尚未安装必要依赖库，正在尝试自动安装基础依赖 (torch, pillow, numpy)...
    pip install torch pillow numpy
    if %errorlevel% neq 0 (
        echo [警告] 依赖自动安装未完成，请手动运行: pip install torch pillow numpy
        echo.
    )
)

:: 4. 启动服务端 (server.py 会自动在浏览器中打开 http://127.0.0.1:8000)
echo [服务] 准备就绪，正在启动 Web 交互式控制台...
echo [就绪] 服务已启动！正在自动打开浏览器: http://127.0.0.1:8000
echo        (在当前窗口按下 Ctrl + C 即可随时停止服务)
echo ================================================================
echo.

python server.py

if %errorlevel% neq 0 (
    echo.
    echo [提示] 服务已停止或遇到异常，按任意键退出窗口...
    pause >nul
)
