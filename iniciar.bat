@echo off
title Vetorizador
cd /d "%~dp0"
echo Verificando dependencias...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Falha ao instalar dependencias. Verifique se o Python esta instalado.
  pause
  exit /b 1
)
python app.py
pause
