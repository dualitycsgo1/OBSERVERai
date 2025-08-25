@echo off
echo Installing CS2 Auto Observer...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python er ikke installeret eller ikke i PATH.
    echo Download Python fra https://python.org
    pause
    exit /b 1
)

echo Python fundet. Installerer dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Fejl ved installation af dependencies.
    echo Prøv at køre som administrator.
    pause
    exit /b 1
)

echo.
echo Installation fuldført!
echo.
echo For at starte programmet:
echo   python cs2_auto_observer.py
echo.
echo Husk at konfigurere CS2 gamestate integration først!
echo Se README.md for instruktioner.
echo.
pause
