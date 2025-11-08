
Write-Host "Starting AQI Dashboard System..." -ForegroundColor Green
Write-Host ""

# Get project root directory
$ProjectRoot = $PWD.Path

# Set PYTHONPATH to project root
$env:PYTHONPATH = $ProjectRoot
Write-Host "Setting PYTHONPATH to: $ProjectRoot" -ForegroundColor Cyan

# Check for conda environment
$condaEnvPath = "D:\envs\aqi"
if (Test-Path $condaEnvPath) {
    Write-Host "Using Conda environment: $condaEnvPath" -ForegroundColor Green
    $activateCmd = "conda activate D:\envs\aqi"
} else {
    Write-Host "Conda environment not found at $condaEnvPath" -ForegroundColor Yellow
    Write-Host "Checking for virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "aqi\Scripts\Activate.ps1") {
        Write-Host "Using virtual environment: aqi" -ForegroundColor Green
        $activateCmd = ".\aqi\Scripts\Activate.ps1"
    } else {
        Write-Host "No environment found. Please create one:" -ForegroundColor Red
        Write-Host "   conda create --prefix D:\envs\aqi python=3.10 -y" -ForegroundColor White
        Write-Host "   conda activate D:\envs\aqi" -ForegroundColor White
        Write-Host "   pip install -r requirements.txt" -ForegroundColor White
        exit 1
    }
}

Write-Host ""

# Start FastAPI in background
Write-Host "Starting FastAPI Backend (Port 8000)..." -ForegroundColor Cyan
$fastapi = Start-Process powershell -ArgumentList "-NoExit", "-Command", "$activateCmd; `$env:PYTHONPATH='$ProjectRoot'; cd '$ProjectRoot'; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload" -PassThru -WindowStyle Normal

Start-Sleep -Seconds 5

# Start Streamlit
Write-Host "Starting Streamlit Dashboard (Port 8501)..." -ForegroundColor Magenta
$streamlit = Start-Process powershell -ArgumentList "-NoExit", "-Command", "$activateCmd; `$env:PYTHONPATH='$ProjectRoot'; cd '$ProjectRoot'; streamlit run dashboard/app.py" -PassThru -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Both services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Yellow
Write-Host "   FastAPI Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "   FastAPI API:   http://localhost:8000" -ForegroundColor White
Write-Host "   Streamlit UI:  http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the services" -ForegroundColor Gray
Write-Host ""

# Wait for user input
Write-Host "Press any key to stop all services..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "Stopping services..." -ForegroundColor Red
Stop-Process -Id $fastapi.Id -ErrorAction SilentlyContinue
Stop-Process -Id $streamlit.Id -ErrorAction SilentlyContinue
Write-Host "Services stopped successfully" -ForegroundColor Green
