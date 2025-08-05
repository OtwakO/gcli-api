@echo off
chcp 65001 > nul
cd /D "%~dp0"
uv run app.py
pause