@echo off
echo Starter CS2 Auto Observer som Administrator...
echo.
echo Dette er nødvendigt for at sende tastatur input til CS2.
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator rettigheder bekræftet.
    echo.
    py cs2_auto_observer_standalone.py
) else (
    echo FEJL: Programmet kører ikke som administrator!
    echo.
    echo Højreklik på denne fil og vælg "Kør som administrator"
    echo eller kør følgende kommando i en administrator terminal:
    echo.
    echo py cs2_auto_observer_standalone.py
    echo.
)

pause
