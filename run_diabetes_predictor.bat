@echo off
echo ============================================================
echo  Medical Diagnosis AI - Diabetes Predictor
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/4] Checking directory structure...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "src" mkdir src
echo     Done!

echo.
echo [2/4] Checking if diabetes_predictor.py exists...
if exist "diabetes_predictor.py" (
    echo     Found in root directory
    set SCRIPT_PATH=diabetes_predictor.py
) else if exist "src\diabetes_predictor.py" (
    echo     Found in src directory
    set SCRIPT_PATH=src\diabetes_predictor.py
) else (
    echo     ERROR: diabetes_predictor.py not found!
    echo     Please make sure the file is in the current directory or src folder
    pause
    exit /b 1
)

echo.
echo [3/4] Installing required packages...
pip install numpy pandas scikit-learn matplotlib seaborn --quiet
echo     Done!

echo.
echo [4/4] Running diabetes predictor...
echo ============================================================
echo.

python %SCRIPT_PATH%

echo.
echo ============================================================
echo  Execution completed!
echo ============================================================
pause