# PowerShell restart script for non-blocking execution

Write-Host "Restarting Conjecture EndPoint App..."

# Kill existing Python processes
Write-Host "Killing existing processes..."
Get-Process python.exe -ErrorAction SilentlyContinue | Stop-Process -Force

# Wait a moment for processes to terminate
Start-Sleep -Seconds 2

# Start endpoint app in background using PowerShell
Write-Host "Starting endpoint app in background..."
Start-Process -FilePath "python" -ArgumentList "src/endpoint_app.py","--host","127.0.0.1","--port","8001" -WindowStyle Hidden

# Give service a moment to start
Write-Host "Giving service time to initialize..."
Start-Sleep -Seconds 5

# Poll until service is ready
Write-Host "Waiting for service to be ready..."
$checkCount = 0
$maxChecks = 20

while ($checkCount -lt $maxChecks) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8001/health" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ EndPoint App is ready at http://127.0.0.1:8001" -ForegroundColor Green
            Write-Host "📚 Documentation available at http://127.0.0.1:8001/docs" -ForegroundColor Green
            Write-Host "Done! Service is running and ready in background." -ForegroundColor Green
            exit 0
        }
    } catch {
        # Service not ready yet
    }
    
    Write-Host "Service not ready yet, waiting... ($($checkCount + 1)/$maxChecks)"
    Start-Sleep -Seconds 3
    $checkCount++
}

Write-Host "⚠️ Timeout: Service did not become ready within expected time" -ForegroundColor Yellow
Write-Host "Service may still be starting in background. Check manually at http://127.0.0.1:8001/health" -ForegroundColor Yellow