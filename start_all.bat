@echo off
echo ========================================================
echo Starting PAIMANA System (Backend + Frontend)
echo ========================================================
start "PAIMANA Backend (Port 8000)" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "PAIMANA Frontend (Port 3000)" cmd /k "cd /d %~dp0frontend && npm run dev"
echo.
echo ========================================================
echo Services launched!
echo - Backend API:  http://localhost:8000 (API Docs: http://localhost:8000/docs)
echo - Frontend UI:  http://localhost:3000
echo - Default login:
echo     Admin:   admin@mospi.gov.in   / Admin@123
echo     Analyst: analyst@mospi.gov.in / Analyst@123
echo ========================================================
