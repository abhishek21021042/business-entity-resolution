@echo off
title Business Entity Resolution - Multi-Machine Inference Hub
color 0B
chcp 65001 > nul

:MENU
cls
echo ===================================================================
echo     ⚡ BUSINESS ENTITY RESOLUTION - DISTRIBUTED EXECUTION HUB ⚡
echo ===================================================================
echo.
echo   Select an option to run on this machine:
echo.
echo   [1] 💻 Laptop 1: Run Part 1 of 3 (Entities 0 to 577,514)
echo   [2] 💻 Laptop 2: Run Part 2 of 3 (Entities 577,515 to 1,155,029)
echo   [3] 💻 Laptop 3: Run Part 3 of 3 (Entities 1,155,030 to 1,732,544)
echo.
echo   [4] 🧩 Merge All Shards into Final Submission (D:\output)
echo   [5] 🚀 Run Full Test Data on THIS single laptop (No splitting)
echo   [6] ❌ Exit
echo.
echo ===================================================================
set /p choice="Enter your choice (1-6): "

REM Detect Python executable (venv or global)
set PYTHON_EXE=..\..\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

if "%choice%"=="1" goto SHARD0
if "%choice%"=="2" goto SHARD1
if "%choice%"=="3" goto SHARD2
if "%choice%"=="4" goto MERGE
if "%choice%"=="5" goto FULL
if "%choice%"=="6" goto EXIT

echo Invalid choice, please try again.
pause
goto MENU

:SHARD0
cls
echo ===================================================================
echo   RUNNING SHARD 1 OF 3 (LAPTOP 1)
echo ===================================================================
echo Test Data : D:\test_data
echo Output    : D:\output
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 0 --test-dir D:\test_data --output-dir D:\output
echo.
echo Execution complete!
pause
goto MENU

:SHARD1
cls
echo ===================================================================
echo   RUNNING SHARD 2 OF 3 (LAPTOP 2)
echo ===================================================================
echo Test Data : D:\test_data
echo Output    : D:\output
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 1 --test-dir D:\test_data --output-dir D:\output
echo.
echo Execution complete!
pause
goto MENU

:SHARD2
cls
echo ===================================================================
echo   RUNNING SHARD 3 OF 3 (LAPTOP 3)
echo ===================================================================
echo Test Data : D:\test_data
echo Output    : D:\output
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 2 --test-dir D:\test_data --output-dir D:\output
echo.
echo Execution complete!
pause
goto MENU

:MERGE
cls
echo ===================================================================
echo   MERGING ALL 3 SHARDS INTO FINAL SUBMISSION
echo ===================================================================
echo Output Directory: D:\output
echo Test Directory  : D:\test_data
echo.
%PYTHON_EXE% merge_shards.py --num-shards 3 --output-dir D:\output --test-dir D:\test_data
echo.
pause
goto MENU

:FULL
cls
echo ===================================================================
echo   RUNNING FULL TEST INFERENCE (1 SHARD / SINGLE MACHINE)
echo ===================================================================
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 1 --shard-id 0 --test-dir D:\test_data --output-dir D:\output
echo.
pause
goto MENU

:EXIT
exit
