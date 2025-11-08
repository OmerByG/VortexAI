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

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [*] pip bulunamadi, yukleniyor...
    python -m ensurepip --upgrade
)

if not exist ".venv\" (
    echo [*] Sanal ortam olusturuluyor...
    python -m venv .venv
    echo [*] Gerekli kutuphaneler yukleniyor...
    pip install --upgrade pip
    pip install -r requirements.txt
)

call .venv\Scripts\activate.bat

echo.
echo [*] Kurulum tamamlandi!
cmd /k