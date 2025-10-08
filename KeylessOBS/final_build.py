#!/usr/bin/env python3
"""
CS2 Observer AI - Final Build Script
Creates a complete standalone distribution ready for deployment
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import zipfile

def create_final_build():
    """Create the final production build"""
    print("CS2 Observer AI - Final Build v4.2.0")
    print("=" * 45)
    
    project_root = Path.cwd()
    build_dir = project_root / "final_build"
    
    # Clean and create build directory
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    print("Step 1: Creating standalone executables...")
    
    # Build main CS2 Observer
    cmd_observer = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name=CS2Observer",
        "--add-data=demo_parser_with_positions.exe;.",
        "--add-data=config.json;.",
        "--add-data=observer_ai_model.pkl;.",
        "--add-data=enhanced_positional_models;enhanced_positional_models", 
        "--add-data=complete_positional_dataset_all_85_demos.json;.",
        "--add-data=complete_kill_analysis.json;.",
        "--add-data=maps;maps",
        "--add-data=training_data;training_data",
        "--hidden-import=numpy",
        "--hidden-import=pandas", 
        "--hidden-import=sklearn",
        "--hidden-import=joblib",
        "--hidden-import=positional_observer_ai",
        "--hidden-import=demo_analyzer",
        "--hidden-import=line_of_sight",
        "--console",
        "cs2_duel_detector.py"
    ]
    
    print("  Building CS2Observer.exe...")
    result = subprocess.run(cmd_observer, cwd=project_root)
    if result.returncode != 0:
        print("  ERROR: Failed to build CS2Observer.exe")
        return False
        
    # Build AI Trainer GUI
    cmd_trainer = [
        sys.executable, "-m", "PyInstaller", 
        "--onefile",
        "--name=CS2AITrainer",
        "--windowed",
        "--add-data=demo_parser_with_positions.exe;.",
        "--add-data=config.json;.",
        "--add-data=observer_ai_model.pkl;.",
        "--add-data=enhanced_positional_models;enhanced_positional_models",
        "--add-data=training_data;training_data",
        "--add-data=complete_positional_dataset_all_85_demos.json;.",
        "--add-data=complete_kill_analysis.json;.",
        "--add-data=maps;maps",
        "--hidden-import=tkinter",
        "--hidden-import=ai_trainer",
        "--hidden-import=numpy",
        "--hidden-import=pandas",
        "--hidden-import=sklearn",
        "--hidden-import=joblib",
        "ai_trainer_gui.py"
    ]
    
    print("  Building CS2AITrainer.exe...")
    result = subprocess.run(cmd_trainer, cwd=project_root)
    if result.returncode != 0:
        print("  ERROR: Failed to build CS2AITrainer.exe")
        return False
        
    print("Step 2: Creating distribution package...")
    
    # Copy executables to build directory
    dist_dir = project_root / "dist"
    shutil.copy2(dist_dir / "CS2Observer.exe", build_dir)
    shutil.copy2(dist_dir / "CS2AITrainer.exe", build_dir)
    
    # Copy additional files
    additional_files = [
        "config.json",
        "config_duel.json", 
        "gamestate_integration_obs_updated.cfg",
        "README.md",
        "INSTALLATION.md"
    ]
    
    for file in additional_files:
        src = project_root / file
        if src.exists():
            shutil.copy2(src, build_dir)
            
    # Create startup scripts
    print("Step 3: Creating startup scripts...")
    
    start_observer = build_dir / "Start_CS2_Observer.bat"
    start_observer.write_text("""@echo off
title CS2 Observer AI
echo Starting CS2 Observer AI...
echo.
echo Make sure CS2 is running with gamestate integration enabled!
echo.
CS2Observer.exe
pause""", encoding='utf-8')

    start_trainer = build_dir / "Start_AI_Trainer.bat"
    start_trainer.write_text("""@echo off
title CS2 AI Trainer
echo Starting CS2 AI Trainer...
echo.
CS2AITrainer.exe""", encoding='utf-8')

    setup_script = build_dir / "Setup_Gamestate_Integration.bat"
    setup_script.write_text("""@echo off
title CS2 Observer AI - Setup
echo CS2 Observer AI - Gamestate Integration Setup
echo ============================================
echo.
set /p cs2_path="Enter your CS2 installation path: "
echo.
if exist "%cs2_path%\\game\\csgo\\cfg" (
    copy "gamestate_integration_obs_updated.cfg" "%cs2_path%\\game\\csgo\\cfg\\"
    echo SUCCESS: Gamestate integration file copied!
) else (
    echo ERROR: CS2 cfg folder not found. Please check the path.
)
echo.
pause""", encoding='utf-8')

    # Create installation guide
    print("Step 4: Creating installation guide...")
    
    install_guide = build_dir / "INSTALLATION_GUIDE.txt" 
    install_guide.write_text("""CS2 Observer AI v4.2.0 - Installation Guide

QUICK START:
1. Extract all files to a folder
2. Run Setup_Gamestate_Integration.bat and enter your CS2 path
3. Launch CS2
4. Run Start_CS2_Observer.bat
5. Enjoy AI-powered camera control!

WHAT'S INCLUDED:
- CS2Observer.exe - Main observer system with AI
- CS2AITrainer.exe - GUI for training the AI with new demos
- Gamestate integration file for CS2
- Startup scripts for easy launching
- All AI models and training data

FEATURES:
- 96.6% AI accuracy trained on 85 professional demos
- Real-time kill prediction and camera switching
- Advanced positional analysis
- Line-of-sight validation
- Continuous learning capability

REQUIREMENTS:
- Windows 10/11 (64-bit)
- Counter-Strike 2 installed
- 4GB RAM minimum

TROUBLESHOOTING:
- Make sure CS2 is running before starting the observer
- Verify gamestate integration file is in CS2 cfg folder
- Check that port 8082 is not blocked by firewall
- Run as administrator if needed

For support and updates, visit the project repository.
Enjoy your AI-powered CS2 observing experience!""", encoding='utf-8')

    print("Step 5: Creating ZIP distribution...")
    
    # Create ZIP file
    zip_path = project_root / f"CS2ObserverAI_v4.2.0_Complete.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in build_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(build_dir)
                zipf.write(file_path, arcname)
                
    file_size = zip_path.stat().st_size / (1024 * 1024)
    
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 35)
    print(f"Build folder: {build_dir}")
    print(f"ZIP package: {zip_path.name} ({file_size:.1f} MB)")
    print("")
    print("DISTRIBUTION CONTENTS:")
    print("- CS2Observer.exe (Main AI observer)")
    print("- CS2AITrainer.exe (AI training GUI)")
    print("- Setup scripts and configuration files")
    print("- Complete installation guide")
    print("- All AI models and training data included")
    print("")
    print("READY FOR DEPLOYMENT!")
    print("Users can download the ZIP, extract, and run immediately.")
    print("No Python, Go, or other dependencies required!")
    
    return True

if __name__ == "__main__":
    success = create_final_build()
    if success:
        print("\nSUCCESS: Production build created!")
        sys.exit(0)
    else:
        print("\nFAILED: Build process failed")
        sys.exit(1)