@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 scripts\bootstrap_gui.py
exit /b %errorlevel%

:use_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python scripts\bootstrap_gui.py
exit /b %errorlevel%

:no_python
echo Cannot start GUI: Python 3.10 or newer was not found. 1>&2
exit /b 1
