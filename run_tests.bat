@echo off
set PYTHON_EXE=C:\ProgramData\anaconda3\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)
"%PYTHON_EXE%" tests\test_esg_pipeline.py
pause
