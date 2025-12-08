#!/bin/bash

echo "Restarting Conjecture EndPoint App..."

# Kill existing Python processes
echo "Killing existing processes..."
pkill -f "python src/endpoint_app.py" || true

# Wait a moment for processes to terminate
sleep 2

# Start endpoint app
echo "Starting endpoint app..."
python src/endpoint_app.py --host 127.0.0.1 --port 8001 &
ENDPOINT_PID=$!

# Poll until service is ready
echo "Waiting for service to be ready..."
while true; do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health | grep -q "200"; then
        echo "✅ EndPoint App is ready at http://127.0.0.1:8001"
        echo "📚 Documentation available at http://127.0.0.1:8001/docs"
        break
    fi
    
    echo "Service not ready yet, waiting..."
    sleep 3
done

echo "Done! Service is running and ready (PID: $ENDPOINT_PID)."