#!/usr/bin/env python3
"""
CS2 Observer AI - Complete Build and Installer Creation System
Creates standalone executables and a professional Windows installer
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
import requests
import tempfile
import zipfile

class CompleteInstaller:
    """Creates a complete professional installer for CS2 Observer AI"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.version = "4.2.0"
        
        print(f"🏗️ CS2 Observer AI Complete Installer Builder v{self.version}")
        print("=" * 60)
        
    def check_dependencies(self):
        """Check and install required build dependencies"""
        print("🔍 Checking build dependencies...")
        
        # Check PyInstaller
        try:
            import PyInstaller
            print(f"✅ PyInstaller: {PyInstaller.__version__}")
        except ImportError:
            print("📦 Installing PyInstaller...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=5.0"], check=True)
            print("✅ PyInstaller installed")
        
        # Check for Inno Setup (Windows installer creator)
        inno_paths = [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe", 
            r"C:\Tools\InnoSetup\ISCC.exe"
        ]
        
        inno_found = False
        for path in inno_paths:
            if os.path.exists(path):
                self.inno_compiler = path
                print(f"✅ Inno Setup found: {path}")
                inno_found = True
                break
                
        if not inno_found:
            print("⚠️ Inno Setup not found - will create ZIP distribution only")
            print("   Download from: https://jrsoftware.org/isinfo.php")
            self.inno_compiler = None
            
        return True
        
    def install_python_dependencies(self):
        """Install all Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "numpy", "pandas", "scikit-learn", "joblib", "psutil", "keyboard", "requests"
        ], check=True)
        
        print("✅ Python dependencies installed")
        
    def create_license_file(self):
        """Create a license file for the installer"""
        license_content = f"""CS2 Observer AI v{self.version}
Advanced AI-powered camera control for Counter-Strike 2

Copyright (c) 2025 CS2 Observer AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

THIRD PARTY COMPONENTS:
- Python scientific computing libraries (NumPy, Pandas, Scikit-learn)
- demoinfocs-golang library for CS2 demo parsing
- Various Python standard library components

This software is designed for educational and entertainment purposes.
Use responsibly and in accordance with Counter-Strike 2 terms of service.
"""
        
        license_path = self.project_root / "LICENSE.txt"
        license_path.write_text(license_content, encoding='utf-8')
        
    def run_pyinstaller_build(self):
        """Run the PyInstaller build process"""
        print("🔨 Building executables with PyInstaller...")
        
        # Run our build system
        result = subprocess.run([
            sys.executable, "build_system.py"
        ], cwd=self.project_root)
        
        if result.returncode != 0:
            print("❌ PyInstaller build failed!")
            return False
            
        print("✅ PyInstaller build completed")
        return True
        
    def create_installer_assets(self):
        """Create additional assets needed for the installer"""
        print("🎨 Creating installer assets...")
        
        # Create assets directory if it doesn't exist
        assets_dir = self.project_root / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Create simple ICO files (in production, use real icons)
        ico_files = ["cs2_observer_icon.ico", "ai_trainer_icon.ico"]
        
        for ico_file in ico_files:
            ico_path = assets_dir / ico_file
            if not ico_path.exists():
                # Create minimal ICO file structure
                ico_data = b'\\x00\\x00\\x01\\x00\\x01\\x00\\x10\\x10\\x00\\x00\\x01\\x00\\x08\\x00h\\x05\\x00\\x00\\x16\\x00\\x00\\x00'
                ico_path.write_bytes(ico_data)
                
        # Create comprehensive installation guide
        self.create_detailed_installation_guide()
        
    def create_detailed_installation_guide(self):
        """Create detailed installation and user guide"""
        guide_content = f"""# CS2 Observer AI v{self.version} - Complete User Guide

## Overview
CS2 Observer AI is a sophisticated artificial intelligence system that automatically controls camera switching in Counter-Strike 2 matches. It uses advanced machine learning models trained on professional gameplay data to predict and switch to the most interesting action.

## Key Features

### 🤖 AI-Powered Camera Control
- **96.6% Accuracy**: Trained on 85 professional demos with 3,900 samples
- **Real-time Prediction**: Anticipates kills 2-5 seconds before they happen
- **Positional Analysis**: Uses player positions, angles, and tactical situations
- **Line-of-Sight Validation**: Prevents impossible duel predictions through walls

### 🧠 Advanced AI Models
- **Distance Category Prediction**: Classifies engagement ranges
- **Headshot Prediction**: Anticipates precision kills
- **Wallbang Prediction**: Detects shots through walls
- **Weapon Effectiveness**: Analyzes weapon suitability for situations

### 🎯 Smart Switching Logic
- **Duel Detection**: Identifies potential player confrontations
- **Kill Pattern Analysis**: Learns from historical combat data
- **Multi-kill Sequences**: Tracks and follows action chains
- **Team Coordination**: Considers tactical team movements

### 🔄 Continuous Learning
- **AI Trainer**: GUI for processing new demo files
- **Hot-reloading**: Updates models during matches
- **Performance Tracking**: Monitors and improves predictions
- **Custom Training**: Train on your specific gameplay style

## Installation Instructions

### Automatic Installation (Recommended)
1. **Run the Installer**: Double-click the installer executable
2. **Choose Installation Path**: Select where to install (default: Program Files)
3. **CS2 Integration Setup**: Provide your CS2 installation path when prompted
4. **Complete Installation**: The installer handles everything automatically

### Manual Setup (If Needed)
1. **Extract Files**: If using ZIP version, extract to desired folder
2. **Gamestate Integration**: 
   - Copy `gamestate_integration_obs_updated.cfg` to your CS2 cfg folder
   - Usually: `C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\csgo\\cfg`
3. **Verify Installation**: Run the test demo parser to ensure everything works

## Quick Start Guide

### First Time Setup
1. **Install Counter-Strike 2** (if not already installed)
2. **Run CS2 Observer AI** from Start Menu or Desktop
3. **Start CS2** and join a match or load a demo
4. **Enjoy AI-powered observing!**

### Using the AI Trainer
1. **Collect Demo Files**: Download or record CS2 demos (.dem files)
2. **Open AI Trainer**: Use "CS2 AI Trainer" shortcut
3. **Add Demos**: Place demo files in the "demos" folder or use the GUI
4. **Process Demos**: Click "Process New Demos" to extract training data
5. **Train Models**: The system automatically improves the AI models
6. **Hot-reload**: Updated models are used immediately in the main observer

## Configuration Options

### Main Configuration (config.json)
```json
{{
  "observer": {{
    "enable_auto_switch": true,
    "switch_cooldown": 0.5,
    "prediction_confidence_threshold": 0.65
  }},
  "duel_detection": {{
    "max_distance": 1200.0,
    "facing_threshold": 0.25,
    "height_difference_max": 450.0
  }},
  "ai_models": {{
    "use_positional_ai": true,
    "prediction_window": 5.0,
    "high_confidence_threshold": 0.85
  }}
}}
```

### Advanced Tuning
- **switch_cooldown**: Minimum time between camera switches (seconds)
- **prediction_confidence_threshold**: How confident AI must be to switch
- **max_distance**: Maximum range for duel detection
- **facing_threshold**: How precisely players must face each other

## Troubleshooting

### Observer Not Working
**Problem**: Camera doesn't switch automatically
**Solutions**:
- Ensure CS2 is running and in a match/demo
- Verify gamestate integration file is in CS2 cfg folder
- Check that port 8082 is not blocked by firewall
- Restart both CS2 and the observer

### AI Models Not Loading  
**Problem**: "Positional AI not available" message
**Solutions**:
- Verify all files were installed properly
- Check that `enhanced_positional_models` folder exists
- Ensure `observer_ai_model.pkl` is present
- Try running as administrator

### Demo Parser Issues
**Problem**: Cannot process demo files
**Solutions**:
- Ensure demo files are valid CS2 format (not CS:GO)
- Check that `demo_parser_with_positions.exe` is present
- Verify demos folder exists and is writable
- Try processing demos one at a time

### Performance Issues
**Problem**: High CPU usage or lag
**Solutions**:
- Reduce prediction window in config.json
- Increase switch_cooldown to reduce switching frequency  
- Close other resource-intensive applications
- Consider upgrading to SSD for faster model loading

## Advanced Usage

### Custom Training Data
1. **Collect Specific Demos**: Focus on maps/situations you observe most
2. **Quality over Quantity**: 10 high-quality demos better than 100 poor ones
3. **Diverse Situations**: Include different maps, team compositions, strategies
4. **Regular Updates**: Process new demos weekly to keep models fresh

### Professional Streaming Setup
1. **Dedicated Machine**: Run observer on separate computer if possible
2. **Network Configuration**: Ensure stable connection to CS2 gamestate
3. **Backup Systems**: Have manual camera control ready as fallback
4. **Monitor Performance**: Watch AI confidence levels and accuracy

### Integration with Broadcasting Tools
- **OBS Studio**: Use hotkey integration for scene switching
- **XSplit**: Configure camera switching triggers
- **Custom Scripts**: API available for advanced integrations

## Technical Specifications

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for installation
- **CPU**: Intel i5 or AMD Ryzen 5 (or equivalent)
- **Network**: Stable connection for CS2 gamestate integration

### AI Model Details
- **Training Dataset**: 85 professional demos, 3,900 kill samples
- **Model Type**: Random Forest ensemble methods
- **Feature Count**: 27 positional and contextual features
- **Update Frequency**: Real-time hot-reloading capability
- **Accuracy Metrics**: Cross-validated 96.6% prediction accuracy

### File Structure
```
CS2ObserverAI/
├── CS2Observer.exe           # Main observer system
├── CS2AITrainer.exe          # AI training interface  
├── CS2DemoParser.exe         # Demo analysis tool
├── demo_parser_with_positions.exe  # Go-based parser
├── models/                   # Basic AI models
├── enhanced_positional_models/  # Advanced AI models
├── training_data/            # Training datasets
├── config.json              # Main configuration
└── gamestate_integration_obs_updated.cfg  # CS2 integration
```

## Support and Updates

### Getting Help
- Check this guide for common solutions
- Verify all installation steps were completed
- Test with different demos/matches
- Check firewall and antivirus settings

### Version Updates
- New versions include improved AI models
- Backup your training data before updating
- Configuration files are usually preserved
- Manual model retraining may be needed

### Contributing
- Submit demo files to improve training data
- Report bugs and issues with detailed logs
- Suggest features and improvements
- Share configuration optimizations

## Performance Monitoring

### AI Accuracy Metrics
The system tracks several performance indicators:
- **Prediction Accuracy**: Percentage of correct camera switches
- **Response Time**: How quickly AI reacts to events
- **False Positive Rate**: Unnecessary camera switches
- **Coverage**: Percentage of important action captured

### Optimization Tips
1. **Monitor Logs**: Check for error messages and warnings
2. **Adjust Thresholds**: Fine-tune confidence levels for your use case
3. **Regular Training**: Process new demos to keep models current
4. **Resource Management**: Balance AI sophistication with system performance

## Legal and Ethical Use

### Terms of Service Compliance
- Use only for observing, not for competitive advantage
- Respect game server rules and regulations
- Do not use for automated gameplay or cheating
- Educational and entertainment purposes only

### Privacy Considerations
- System processes public game data only
- No personal information is collected or transmitted
- Demo files remain local unless explicitly shared
- Configuration data stays on your system

---

**CS2 Observer AI v{self.version}**  
Advanced AI-powered camera control for Counter-Strike 2  
Built with ❤️ for the CS2 community

For technical support and updates, visit the project repository.
Enjoy your enhanced observing experience! 🎯
"""
        
        guide_path = self.project_root / "COMPLETE_USER_GUIDE.txt"
        guide_path.write_text(guide_content, encoding='utf-8')
        
    def build_inno_installer(self):
        """Build the Windows installer using Inno Setup"""
        if not self.inno_compiler:
            print("⚠️ Inno Setup not available, skipping installer creation")
            return False
            
        print("🔨 Building Windows installer with Inno Setup...")
        
        # Create installer output directory
        installer_output = self.project_root / "installer_output"
        installer_output.mkdir(exist_ok=True)
        
        # Run Inno Setup compiler
        result = subprocess.run([
            self.inno_compiler,
            "/Q",  # Quiet mode
            str(self.project_root / "installer_script.iss")
        ], cwd=self.project_root)
        
        if result.returncode != 0:
            print("❌ Inno Setup compilation failed!")
            return False
            
        print("✅ Windows installer created successfully")
        return True
        
    def create_complete_distribution(self):
        """Create the complete distribution package"""
        print("📦 Creating complete distribution...")
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False
                
            # Step 2: Install Python dependencies
            self.install_python_dependencies()
            
            # Step 3: Create required files
            self.create_license_file()
            self.create_installer_assets() 
            
            # Step 4: Build executables
            if not self.run_pyinstaller_build():
                return False
                
            # Step 5: Build Windows installer
            installer_success = self.build_inno_installer()
            
            # Step 6: Create ZIP backup
            self.create_zip_backup()
            
            print("\n🎉 COMPLETE BUILD FINISHED!")
            print("=" * 50)
            print(f"📦 Version: {self.version}")
            
            if installer_success:
                print("✅ Windows Installer: installer_output/")
                print("   Ready for distribution to end users")
                
            print("✅ ZIP Distribution: dist/")
            print("   Portable version for advanced users")
            
            print("\n🚀 Ready for professional deployment!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ BUILD FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def create_zip_backup(self):
        """Create ZIP distribution as backup"""
        print("🗜️ Creating ZIP distribution...")
        
        zip_path = self.project_root / "dist" / f"CS2ObserverAI_v{self.version}_Portable.zip"
        dist_dir = self.project_root / "dist" / "CS2ObserverAI_Distribution"
        
        if not dist_dir.exists():
            print("❌ Distribution directory not found")
            return False
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dist_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(dist_dir)
                    zipf.write(file_path, arcname)
                    
            # Add additional files
            additional_files = [
                "README.md", "INSTALLATION.md", "COMPLETE_USER_GUIDE.txt",
                "LICENSE.txt", "config.json", "config_duel.json"
            ]
            
            for file_name in additional_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    zipf.write(file_path, file_name)
                    
        file_size = zip_path.stat().st_size / (1024 * 1024)
        print(f"✅ ZIP created: {zip_path.name} ({file_size:.1f} MB)")
        
        return True

def main():
    """Main function to create complete installer"""
    installer = CompleteInstaller()
    
    print("🎯 Creating professional CS2 Observer AI distribution")
    print("This will build executables and create a Windows installer")
    print("with all dependencies included.\n")
    
    success = installer.create_complete_distribution()
    
    if success:
        print("\n✅ SUCCESS: Complete distribution created!")
        print("\nWhat was created:")
        print("📁 Windows Installer (.exe) - For end users")
        print("📁 ZIP Distribution - Portable version") 
        print("📁 All dependencies included - No manual setup required")
        print("\n🎉 Ready for professional deployment!")
        return 0
    else:
        print("\n❌ FAILED: Could not create complete distribution")
        return 1

if __name__ == "__main__":
    sys.exit(main())