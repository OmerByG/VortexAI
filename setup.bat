@echo off
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

cd /d "%~dp0"
chcp 65001 >nul
title VortexAI Kurulum

echo ============================================================
echo [*] VortexAI Kurulum Basliyor...
echo ============================================================

call .venv\Scripts\activate.bat

echo.
echo [*] Kurulum tamamlandi!
cmd /k