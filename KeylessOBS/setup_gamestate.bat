@echo off
echo CS2 Gamestate Integration Setup
echo ================================
echo.

set "CS2_CFG_PATH=C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"
set "SOURCE_FILE=%~dp0gamestate_integration_obs_updated.cfg"

echo Checker CS2 installation...
if not exist "%CS2_CFG_PATH%" (
    echo FEJL: CS2 cfg mappe ikke fundet!
    echo Forventet placering: %CS2_CFG_PATH%
    echo.
    echo Prøv at finde din Steam installation:
    echo - Højreklik på CS2 i Steam
    echo - Properties ^> Installed Files ^> Browse
    echo - Naviger til game\csgo\cfg\
    echo.
    pause
    exit /b 1
)

echo CS2 cfg mappe fundet: %CS2_CFG_PATH%
echo.

if not exist "%SOURCE_FILE%" (
    echo FEJL: gamestate_integration_obs_updated.cfg ikke fundet!
    echo Sørg for at køre denne fil fra KeylessOBS mappen.
    pause
    exit /b 1
)

echo Kopierer gamestate fil...
copy "%SOURCE_FILE%" "%CS2_CFG_PATH%\gamestate_integration_obs.cfg"

if errorlevel 1 (
    echo FEJL ved kopiering! Prøv at køre som administrator.
    pause
    exit /b 1
)

echo.
echo ✓ Gamestate fil kopieret til CS2!
echo.
echo Næste skridt:
echo 1. Genstart CS2 hvis det kører
echo 2. Gå ind som observer i en match
echo 3. Programmet skulle nu modtage data
echo.
pause
