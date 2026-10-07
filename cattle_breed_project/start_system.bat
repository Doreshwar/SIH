@echo off
echo Starting Cattle Breed Identification System...
echo ==========================================

cd /d "%~dp0"

echo Setting up backend...
cd backend
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt

echo Starting backend...
start /B python app.py

echo Setting up frontend...
cd ..\cattle-sense-main
call npm install

echo Starting frontend...
start /B npm run dev

echo System ready!
echo Frontend: http://localhost:5173
echo Backend: http://localhost:5000
pause