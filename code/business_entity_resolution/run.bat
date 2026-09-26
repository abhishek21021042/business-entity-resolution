@echo off
title Business Entity Resolution - Multi-Machine Inference Hub

REM Always run from the directory where this bat file lives
cd /d "%~dp0"

REM Detect Python executable (venv or global)
set PYTHON_EXE=..\..\venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=..\..\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

:MENU
cls
echo ===================================================================
echo     BUSINESS ENTITY RESOLUTION - DISTRIBUTED EXECUTION HUB
echo ===================================================================
echo.
echo   [1] Launch Modern Web Dashboard (Browser UI with Live Progress)
echo.
echo   --- Terminal Execution (Auto-detects D:\ or Project Folder) ---
echo   [2] Laptop 1: Run Part 1 of 3 (Entities 0 to 577,514)
echo   [3] Laptop 2: Run Part 2 of 3 (Entities 577,515 to 1,155,029)
echo   [4] Laptop 3: Run Part 3 of 3 (Entities 1,155,030 to 1,732,544)
echo.
echo   [5] Merge All Shards into Final Submission & Validate
echo   [6] Run Full Test Data on THIS single laptop (No splitting)
echo   [7] Exit
echo.
echo ===================================================================
set /p choice=Enter your choice (1-7): 

if "%choice%"=="1" goto WEB
if "%choice%"=="2" goto SHARD0
if "%choice%"=="3" goto SHARD1
if "%choice%"=="4" goto SHARD2
if "%choice%"=="5" goto MERGE
if "%choice%"=="6" goto FULL
if "%choice%"=="7" goto EXIT

echo Invalid choice, please try again.
pause
goto MENU

:WEB
cls
echo ===================================================================
echo   LAUNCHING INTERACTIVE WEB DASHBOARD
echo ===================================================================
echo Opening in your browser...
%PYTHON_EXE% web_app.py
pause
goto MENU

:SHARD0
cls
echo ===================================================================
echo   RUNNING SHARD 1 OF 3 (LAPTOP 1)
echo ===================================================================
echo Auto-detecting dataset and cache...
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 0
echo.
echo Execution complete!
pause
goto MENU

:SHARD1
cls
echo ===================================================================
echo   RUNNING SHARD 2 OF 3 (LAPTOP 2)
echo ===================================================================
echo Auto-detecting dataset and cache...
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 1
echo.
echo Execution complete!
pause
goto MENU

:SHARD2
cls
echo ===================================================================
echo   RUNNING SHARD 3 OF 3 (LAPTOP 3)
echo ===================================================================
echo Auto-detecting dataset and cache...
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 3 --shard-id 2
echo.
echo Execution complete!
pause
goto MENU

:MERGE
cls
echo ===================================================================
echo   MERGING ALL 3 SHARDS INTO FINAL SUBMISSION
echo ===================================================================
echo.
%PYTHON_EXE% merge_shards.py --num-shards 3
echo.
pause
goto MENU

:FULL
cls
echo ===================================================================
echo   RUNNING FULL TEST INFERENCE (1 SHARD / SINGLE MACHINE)
echo ===================================================================
echo.
%PYTHON_EXE% src\shard_infer.py --num-shards 1 --shard-id 0
echo.
pause
goto MENU

:EXIT
exit
