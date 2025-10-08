@echo off
title CS2 Observer AI v1.0
color 0A
echo ========================================
echo       CS2 Observer AI - Starting...
echo ========================================
echo.
echo Make sure CS2 is running with gamestate integration enabled!
echo.
echo Starting the AI-powered camera system...
echo.
CS2Observer.exe
if errorlevel 1 (
    echo.
    echo ERROR: CS2Observer failed to start!
    echo Check that CS2 is running and gamestate integration is configured.
    echo.
    pause
)