@echo off
REM NEURON v2.0 - Self-Learning AI Agent Launcher

echo ========================================
echo  NEURON v2.0 - Self-Learning AI Agent
echo ========================================
echo.

REM Check if backend is running
echo [1/3] Checking backend...
python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=2)" 2>nul
if %errorlevel% equ 0 (
    echo  ✓ Backend is running
) else (
    echo  ✗ Backend not running
    echo.
    echo  Starting backend...
    start "NEURON Backend" cmd /c "cd /d %~dp0..\backend && python main.py"
    echo  Waiting for backend to start...
    timeout /t 5 /nobreak > nul
)

REM Install frontend dependencies if needed
if not exist node_modules (
    echo [2/3] Installing frontend dependencies...
    npm install
    echo  ✓ Dependencies installed
) else (
    echo [2/3] Dependencies already installed
)

REM Start frontend
echo [3/3] Starting frontend...
start "NEURON Frontend" cmd /c "npm run dev"

echo.
echo ========================================
echo  NEURON v2.0 is starting!
echo ========================================
echo.
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:3000
echo  API Docs: http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop both services
echo ========================================

pause
