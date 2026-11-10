@echo off
title ESG BERT Automated Intelligence Dashboard
echo ========================================================
echo Starting ESG BERT Automated Intelligence Dashboard...
echo ========================================================
set PYTHON_EXE=C:\ProgramData\anaconda3\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" -m streamlit run app.py
pause
