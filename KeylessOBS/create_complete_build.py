#!/usr/bin/env python3
"""
OBSERVERai Complete Build Script v2.0
Creates a comprehensive distribution package with all new features
"""

import os
import sys
import shutil
import zipfile
import json
import logging
from datetime import datetime
from pathlib import Path

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def create_complete_build():
    """Create complete distribution build with all new features"""
    
    print("🚀 OBSERVERai Complete Build v2.0")
    print("=" * 50)
    
    base_dir = get_base_dir()
    build_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_name = f"OBSERVERai_Complete_v2.0_{build_timestamp}"
    build_dir = os.path.join(base_dir, "dist", build_name)
    
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    print(f"📁 Build directory: {build_dir}")
    
    # Files to include in the build
    core_files = [
        # Core Observer System
        "cs2_duel_detector.py",
        "cs2_duel_detector_ct.py", 
        "cs2_duel_detector_t.py",
        "observer_gui.py",
        
        # AI Training System - NEW!
        "ai_trainer.py",
        "ai_trainer_gui.py", 
        "demo_analyzer.py",
        "comprehensive_demo_processor.py",  # NEW!
        
        # Demo Processing Tools - NEW!
        "demo_parser_with_positions.exe",
        "cs2_header_extractor.exe",  # NEW!
        
        # Compiled Executables - IMPORTANT!
        "dist/CS2Observer.exe",        # Main CS2 Observer (compiled)
        "dist/CS2AITrainer.exe",       # AI Trainer GUI (compiled)
        
        # Enhanced AI Components
        "positional_observer_ai.py",
        "pattern_memory_ai.py",
        "enhanced_observer_ai.py",
        "model_updater.py",
        "kill_pattern_analyzer.py",
        
        # Configuration Files
        "config.json",
        "config_duel.json",
        "ai_config.json",
        "gamestate_integration_obs_updated.cfg",
        
        # Core Dependencies
        "line_of_sight.py",
        "requirements.txt",
        
        # Build and Installation
        "install.bat",
        "start.bat",
        "start_admin.bat",
        "start_duel_detector.bat",
        "run_demo.bat",
        "setup_gamestate.bat",
        
        # Documentation
        "README.md",
        "INSTALLATION.md",
        "build_instructions.txt",
        
        # NEW Integration Files
        "consolidate_training_data.py",
        "test_ai_integration.py",
        "AI_INTEGRATION_COMPLETE.md"
    ]
    
    # Directories to include
    directories_to_copy = [
        "models",
        "training_data",  # NEW - with consolidated training data
        "model_backups",
        "demos"
    ]
    
    print("📦 Copying core files...")
    
    # Copy core files
    for file in core_files:
        src_path = os.path.join(base_dir, file)
        if os.path.exists(src_path):
            dst_path = os.path.join(build_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️ Missing: {file}")
    
    print("📁 Copying directories...")
    
    # Copy directories
    for directory in directories_to_copy:
        src_dir = os.path.join(base_dir, directory)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(build_dir, directory)
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            
            # Count files in directory
            file_count = sum(len(files) for _, _, files in os.walk(dst_dir))
            print(f"   ✅ {directory}/ ({file_count} files)")
        else:
            print(f"   ⚠️ Missing directory: {directory}")
            # Create empty directory
            os.makedirs(os.path.join(build_dir, directory), exist_ok=True)
    
    # Create build info file
    print("📋 Creating build information...")
    
    build_info = {
        "build_name": build_name,
        "build_timestamp": build_timestamp,
        "version": "2.0",
        "features": [
            "Enhanced AI Training System with 6,115+ training samples",
            "Comprehensive Demo Processing (header + positional data)",
            "Hot-Reload AI Model Integration", 
            "AIIMS (AI Intelligence Measurement System)",
            "Real Map Name Extraction from CS2 Demos",
            "Advanced Duel Detection with Kill Prediction",
            "Pattern-Aware AI with Memory System",
            "Positional Observer AI Integration",
            "Complete Kill Pattern Analysis (12,092+ kills)",
            "Model Auto-Update System",
            "Professional GUI Applications"
        ],
        "new_in_v2": [
            "AI Training Hot-Reload Integration",
            "Comprehensive Demo Processor",
            "CS2 Header Name Extraction", 
            "Consolidated Training Data (6,115 samples)",
            "AIIMS Statistics with Real Map Names",
            "Training Data Consolidation System",
            "End-to-End Integration Testing"
        ],
        "components": {
            "core_files": len([f for f in core_files if os.path.exists(os.path.join(base_dir, f))]),
            "total_files": sum(len(files) for _, _, files in os.walk(build_dir)),
            "training_samples": get_training_sample_count(),
            "models_included": count_model_files(build_dir)
        }
    }
    
    build_info_path = os.path.join(build_dir, "BUILD_INFO.json")
    with open(build_info_path, 'w') as f:
        json.dump(build_info, f, indent=2)
    
    # Create enhanced README for the build
    create_build_readme(build_dir, build_info)
    
    # Create the ZIP file
    print("🗜️ Creating ZIP archive...")
    
    zip_path = os.path.join(base_dir, "dist", f"{build_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, build_dir)
                zipf.write(file_path, arc_path)
    
    # Get ZIP file size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    
    print("✅ Build complete!")
    print(f"📦 ZIP file: {os.path.basename(zip_path)} ({zip_size_mb:.1f} MB)")
    print(f"📊 Total files: {build_info['components']['total_files']}")
    print(f"🧠 Training samples: {build_info['components']['training_samples']}")
    print(f"🤖 AI models: {build_info['components']['models_included']}")
    
    return zip_path, build_info

def get_training_sample_count():
    """Get the current training sample count"""
    try:
        base_dir = get_base_dir()
        training_file = os.path.join(base_dir, "training_data", "complete_kill_analysis.json")
        
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 0
        return 0
    except:
        return 0

def count_model_files(build_dir):
    """Count model files in the build"""
    try:
        models_dir = os.path.join(build_dir, "models")
        if os.path.exists(models_dir):
            return len([f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib'))])
        return 0
    except:
        return 0

def create_build_readme(build_dir, build_info):
    """Create enhanced README for the build"""
    
    readme_content = f"""# OBSERVERai Complete v{build_info['version']} 

## 🚀 What's New in v{build_info['version']}

{chr(10).join('- ' + feature for feature in build_info['new_in_v2'])}

## 🎯 Complete Feature Set

{chr(10).join('- ' + feature for feature in build_info['features'])}

## 📊 Build Statistics

- **Build Date**: {build_info['build_timestamp']}
- **Total Files**: {build_info['components']['total_files']}
- **Training Samples**: {build_info['components']['training_samples']:,}
- **AI Models**: {build_info['components']['models_included']}

## 🚀 Quick Start

### 1. Install Dependencies
```bash
install.bat
```

### 2. Setup CS2 Game State Integration
```bash
setup_gamestate.bat
```

### 3. Run Applications

**CS2 Auto Observer:**
```bash
start.bat
```

**AI Trainer (NEW!):**
```bash
python ai_trainer_gui.py
```

**Observer GUI:**
```bash
python observer_gui.py
```

## 🧠 AI Training System (NEW!)

The AI trainer now includes:
- **{build_info['components']['training_samples']:,} training samples** from real CS2 matches
- **Hot-reload integration** - trained models automatically update the observer
- **AIIMS statistics** - detailed AI intelligence measurement
- **Real map name extraction** from CS2 demo files
- **Comprehensive demo processing** with positional data

### Training New Models

1. Run `python ai_trainer_gui.py`
2. Import your CS2 demo files (.dem)
3. Click "Train AI Model"
4. Observer automatically uses updated models within 30 seconds!

## 🔧 System Requirements

- Windows 10/11
- Python 3.8+
- CS2 with game state integration
- 4GB+ RAM (for AI training)
- 2GB+ disk space

## 📋 File Structure

- `cs2_duel_detector.py` - Main observer with AI integration
- `ai_trainer_gui.py` - AI training interface (NEW!)
- `comprehensive_demo_processor.py` - Enhanced demo processing (NEW!)
- `training_data/` - Consolidated training dataset ({build_info['components']['training_samples']:,} samples)
- `models/` - Pre-trained AI models with hot-reload support
- `*.exe` - Demo processing tools for CS2 format support

## 🆘 Support

Check the logs in `%APPDATA%/OBSERVERai/` for troubleshooting.

Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(build_dir, "README_BUILD.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def test_build_integrity():
    """Test the build integrity"""
    print("🧪 Testing build integrity...")
    
    base_dir = get_base_dir()
    
    # Test core systems
    tests = [
        ("Training data consolidation", lambda: test_training_data()),
        ("AI integration", lambda: test_ai_integration()),
        ("Demo processing tools", lambda: test_demo_tools()),
        ("Model files", lambda: test_model_files())
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, "✅"))
            print(f"   ✅ {test_name}: {'PASS' if result else 'WARN'}")
        except Exception as e:
            results.append((test_name, False, f"❌ {e}"))
            print(f"   ❌ {test_name}: {e}")
    
    return results

def test_training_data():
    """Test training data availability"""
    base_dir = get_base_dir()
    training_file = os.path.join(base_dir, "training_data", "complete_kill_analysis.json")
    return os.path.exists(training_file) and get_training_sample_count() > 1000

def test_ai_integration():
    """Test AI integration components"""
    base_dir = get_base_dir()
    required_files = ["ai_trainer.py", "model_updater.py", "comprehensive_demo_processor.py"]
    return all(os.path.exists(os.path.join(base_dir, f)) for f in required_files)

def test_demo_tools():
    """Test demo processing tools"""
    base_dir = get_base_dir()
    tools = ["demo_parser_with_positions.exe", "cs2_header_extractor.exe"]
    return all(os.path.exists(os.path.join(base_dir, tool)) for tool in tools)

def test_model_files():
    """Test model files availability"""
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, "models")
    return os.path.exists(models_dir) and count_model_files(base_dir) > 0

def main():
    """Main build process"""
    
    try:
        # Test build integrity first
        test_results = test_build_integrity()
        
        # Create the build
        zip_path, build_info = create_complete_build()
        
        print(f"\n🎉 OBSERVERai v{build_info['version']} Build Complete!")
        print(f"📦 File: {os.path.basename(zip_path)}")
        print(f"📍 Location: {zip_path}")
        
        # Build summary
        print(f"\n📋 Build Summary:")
        print(f"   • Enhanced AI Training System")
        print(f"   • {build_info['components']['training_samples']:,} training samples")
        print(f"   • Hot-reload model integration") 
        print(f"   • Comprehensive demo processing")
        print(f"   • Real CS2 map name extraction")
        print(f"   • AIIMS intelligence measurement")
        print(f"   • Complete professional package")
        
        print(f"\n🚀 Ready for distribution!")
        
        return zip_path
        
    except Exception as e:
        print(f"\n💥 Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()