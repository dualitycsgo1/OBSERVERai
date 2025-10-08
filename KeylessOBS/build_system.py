#!/usr/bin/env python3
"""
CS2 Observer AI - Complete Build System
Creates a standalone distribution with all dependencies
"""

import os
import sys
import shutil
import subprocess
import json
import zipfile
from pathlib import Path
import urllib.request
import tempfile

class CS2ObserverBuilder:
    """Complete build system for CS2 Observer AI"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.temp_dir = Path(tempfile.gettempdir()) / "cs2_observer_build"
        
        # Build configuration
        self.app_name = "CS2Observer AI"
        self.app_version = "4.2.0"
        self.app_description = "Advanced CS2 Observer with AI-powered camera control"
        
        print(f"🏗️ CS2 Observer AI Builder v{self.app_version}")
        print("=" * 50)
        
    def setup_build_environment(self):
        """Set up the build environment"""
        print("📦 Setting up build environment...")
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check if PyInstaller is installed
        try:
            import PyInstaller
            print(f"✅ PyInstaller found: {PyInstaller.__version__}")
        except ImportError:
            print("❌ PyInstaller not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("✅ PyInstaller installed")
            
        # Check for UPX (optional, for smaller executables)
        try:
            subprocess.run(["upx", "--version"], capture_output=True, check=True)
            print("✅ UPX found (will compress executables)")
        except:
            print("⚠️ UPX not found (executables will be larger)")
            
    def verify_dependencies(self):
        """Verify all dependencies are present"""
        print("🔍 Verifying dependencies...")
        
        required_files = [
            "cs2_duel_detector.py",
            "ai_trainer_gui.py", 
            "complete_demo_parser.py",
            "demo_parser_with_positions.exe",
            "config.json",
            "observer_ai_model.pkl",
            "complete_positional_dataset_all_85_demos.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
                
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
            
        print("✅ All dependencies verified")
        return True
        
    def create_runtime_assets(self):
        """Create additional runtime assets"""
        print("🎨 Creating runtime assets...")
        
        assets_dir = self.project_root / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Create basic ICO files (placeholder)
        ico_files = [
            "cs2_observer_icon.ico",
            "ai_trainer_icon.ico"
        ]
        
        for ico_file in ico_files:
            ico_path = assets_dir / ico_file
            if not ico_path.exists():
                # Create a minimal ICO file (placeholder)
                # In production, you'd use actual icon files
                ico_path.write_bytes(b'\\x00\\x00\\x01\\x00\\x01\\x00\\x10\\x10\\x00\\x00\\x01\\x00\\x08\\x00h\\x05\\x00\\x00\\x16\\x00\\x00\\x00')
                
        # Create startup scripts
        self.create_startup_scripts()
        
    def create_startup_scripts(self):
        """Create Windows batch files for easy startup"""
        print("📝 Creating startup scripts...")
        
        scripts = {
            "Start_CS2_Observer.bat": """@echo off
title CS2 Observer AI - Main System
echo Starting CS2 Observer AI...
echo.
echo Make sure CS2 is running with gamestate integration enabled!
echo Place the gamestate_integration_obs_updated.cfg file in your CS2 cfg folder.
echo.
pause
CS2Observer.exe
pause""",
            
            "Start_AI_Trainer.bat": """@echo off
title CS2 Observer AI - Training System
echo Starting AI Trainer...
echo.
echo Use this to train the AI with new demo files.
echo Place demo files in the 'demos' folder.
echo.
CS2AITrainer.exe""",
            
            "Setup_Gamestate_Integration.bat": """@echo off
title CS2 Observer AI - Setup
echo CS2 Observer AI - Gamestate Integration Setup
echo ============================================
echo.
echo This will copy the gamestate integration file to your CS2 cfg folder.
echo.
set /p cs2_path="Enter your CS2 installation path (e.g., C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive): "
echo.
if exist "%cs2_path%\\game\\csgo\\cfg" (
    copy "gamestate_integration_obs_updated.cfg" "%cs2_path%\\game\\csgo\\cfg\\"
    echo ✅ Gamestate integration file copied successfully!
    echo.
    echo Next steps:
    echo 1. Start CS2
    echo 2. Run Start_CS2_Observer.bat
    echo 3. The observer will automatically control your camera!
) else (
    echo ❌ CS2 cfg folder not found. Please check the path.
    echo Make sure you entered the correct CS2 installation directory.
)
echo.
pause""",
            
            "Test_Demo_Parser.bat": """@echo off
title CS2 Observer AI - Demo Parser Test
echo Testing Demo Parser...
echo.
CS2DemoParser.exe
pause"""
        }
        
        for script_name, script_content in scripts.items():
            script_path = self.project_root / script_name
            script_path.write_text(script_content, encoding='utf-8')
            
    def build_executables(self):
        """Build all executables using PyInstaller"""
        print("🔨 Building executables...")
        
        # Run PyInstaller with our spec file
        spec_file = self.project_root / "build_spec.spec"
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_file)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode != 0:
            print("❌ Build failed!")
            return False
            
        print("✅ Build completed successfully")
        return True
        
    def package_distribution(self):
        """Package the complete distribution"""
        print("📦 Packaging distribution...")
        
        dist_source = self.dist_dir / "CS2ObserverAI_Distribution"
        if not dist_source.exists():
            print("❌ Distribution folder not found!")
            return False
            
        # Create final package directory
        package_name = f"CS2ObserverAI_v{self.app_version}"
        package_dir = self.dist_dir / package_name
        
        if package_dir.exists():
            shutil.rmtree(package_dir)
            
        # Copy distribution
        shutil.copytree(dist_source, package_dir)
        
        # Copy additional files
        additional_files = [
            "README.md",
            "INSTALLATION.md", 
            "config.json",
            "config_duel.json",
            "gamestate_integration_obs_updated.cfg",
            "Start_CS2_Observer.bat",
            "Start_AI_Trainer.bat", 
            "Setup_Gamestate_Integration.bat",
            "Test_Demo_Parser.bat"
        ]
        
        for file in additional_files:
            src = self.project_root / file
            if src.exists():
                dst = package_dir / file
                if src.is_file():
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst)
                    
        # Create installation guide
        self.create_installation_guide(package_dir)
        
        # Create ZIP archive
        zip_path = self.dist_dir / f"{package_name}.zip"
        self.create_zip_archive(package_dir, zip_path)
        
        print(f"✅ Package created: {package_name}")
        print(f"📁 Location: {package_dir}")
        print(f"📦 ZIP archive: {zip_path}")
        
        return True
        
    def create_installation_guide(self, package_dir):
        """Create a comprehensive installation guide"""
        guide_content = f"""# CS2 Observer AI v{self.app_version} - Installation Guide

## What is CS2 Observer AI?
Advanced AI-powered camera control system for Counter-Strike 2 that automatically switches to the most interesting player during matches.

## Features
✅ AI-powered camera switching with 96.6% accuracy
✅ Positional analysis using 85-demo trained dataset
✅ Real-time duel prediction and kill anticipation
✅ Line-of-sight validation to prevent impossible duels
✅ AI trainer for continuous model improvement
✅ Hot-reloading model updates during matches

## Quick Installation (5 minutes)

### Step 1: Extract Files
Extract all files to a folder of your choice (e.g., C:\\CS2ObserverAI)

### Step 2: Setup CS2 Integration
1. Double-click "Setup_Gamestate_Integration.bat"
2. Enter your CS2 installation path when prompted
3. The script will automatically copy the required configuration file

### Step 3: Start the System
1. Launch Counter-Strike 2
2. Double-click "Start_CS2_Observer.bat"
3. The AI observer will start automatically!

## What's Included
- **CS2Observer.exe** - Main observer system
- **CS2AITrainer.exe** - AI training interface (GUI)
- **CS2DemoParser.exe** - Demo analysis tool
- **Pre-trained AI models** - 85-demo dataset with 3,900 samples
- **Configuration files** - Ready-to-use settings
- **Startup scripts** - Easy launch shortcuts

## Advanced Usage

### AI Training
1. Place CS2 demo files (.dem) in the "demos" folder
2. Run "Start_AI_Trainer.bat" 
3. Click "Process New Demos" to improve the AI
4. Models are automatically updated in real-time

### Configuration
Edit config.json to customize:
- Camera switching sensitivity
- Duel detection parameters  
- AI prediction thresholds
- Keyboard shortcuts

### Demo Analysis
- Use "Test_Demo_Parser.bat" to analyze demo files
- Supports CS2 demo format with positional data
- Extracts kill events, player positions, and tactical data

## Troubleshooting

### Observer Not Working
- Make sure CS2 is running
- Verify gamestate integration file is in CS2 cfg folder
- Check that port 8082 is not blocked by firewall

### AI Models Not Loading
- Ensure all files were extracted properly
- Check that enhanced_positional_models folder exists
- Verify observer_ai_model.pkl is present

### Demo Parser Issues
- Make sure demo files are valid CS2 format
- Check that demo_parser_with_positions.exe is present
- Verify demos folder exists and is accessible

## System Requirements
- Windows 10/11 (64-bit)
- 4GB RAM minimum, 8GB recommended
- 2GB free disk space
- Counter-Strike 2 installed
- Active internet connection for initial setup

## Support
For issues and updates, check the project repository or contact support.

## Version {self.app_version} Features
- Complete standalone installation (no dependencies required)
- Enhanced positional AI with 96.6% accuracy
- Real-time model hot-reloading
- Comprehensive demo parser with Go backend
- Professional GUI for AI training
- Automatic dependency management

Enjoy your AI-powered CS2 observing experience! 🎯
"""
        
        guide_path = package_dir / "INSTALLATION_GUIDE.txt"
        guide_path.write_text(guide_content, encoding='utf-8')
        
    def create_zip_archive(self, source_dir, zip_path):
        """Create ZIP archive of the distribution"""
        print(f"🗜️ Creating ZIP archive: {zip_path.name}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)
                    
        file_size = zip_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ ZIP created: {file_size:.1f} MB")
        
    def cleanup(self):
        """Clean up temporary files"""
        print("🧹 Cleaning up...")
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
        # Clean up build artifacts
        build_artifacts = ["build", "__pycache__"]
        for artifact in build_artifacts:
            artifact_path = self.project_root / artifact
            if artifact_path.exists():
                shutil.rmtree(artifact_path)
                
        print("✅ Cleanup completed")
        
    def build_complete_distribution(self):
        """Build the complete distribution package"""
        print(f"🚀 Building {self.app_name} v{self.app_version}")
        print("=" * 60)
        
        try:
            # Step 1: Setup
            self.setup_build_environment()
            
            # Step 2: Verify dependencies  
            if not self.verify_dependencies():
                return False
                
            # Step 3: Create assets
            self.create_runtime_assets()
            
            # Step 4: Build executables
            if not self.build_executables():
                return False
                
            # Step 5: Package distribution
            if not self.package_distribution():
                return False
                
            # Step 6: Cleanup
            self.cleanup()
            
            print("\n🎉 BUILD COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📦 Package: CS2ObserverAI_v{self.app_version}")
            print(f"📁 Location: {self.dist_dir}")
            print("\n🎯 Ready for distribution!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ BUILD FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main build function"""
    builder = CS2ObserverBuilder()
    success = builder.build_complete_distribution()
    
    if success:
        print("\n✅ Build completed successfully!")
        return 0
    else:
        print("\n❌ Build failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())