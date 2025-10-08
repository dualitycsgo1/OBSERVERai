@echo off
title CS2 AI Trainer v1.0
color 0B
echo ========================================
echo       CS2 AI Trainer - Starting...
echo ========================================
echo.
echo Train your AI with demo files for better performance!
echo.
echo Current training statistics:
echo - Current training samples: 1,627 kills from 12 demos
echo - Historical dataset: 3,900 kills from 85 demos  
echo - Total: 5,527 training samples across 97 demos
echo.
echo Starting AI Trainer GUI...
if not exist "model_backups" mkdir model_backups

echo Launching AI Trainer GUI...
echo.
echo Instructions:
echo 1. Add demo files (.dem) using the "Add Demo Files" button
echo 2. Or copy demo files to the "demos" folder and click "Scan Demos Folder"
echo 3. Click "Start Training" to improve your AI
echo 4. Monitor progress in the training log
echo 5. The updated AI will automatically be used by the observer system
echo.

python ai_trainer_gui.py

if errorlevel 1 (
    echo.
    echo GUI failed to start, trying CLI mode...
    echo.
    python ai_trainer_gui.py --cli
)

pause