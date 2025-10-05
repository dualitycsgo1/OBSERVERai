# 🎯 OBSERVERai - Intelligent CS2 Auto Observer v2.0

**The most advanced AI-powered Counter-Strike 2 observer system that predicts kills before they happen.**

OBSERVERai is a revolutionary AI system that automatically switches your CS2 observer camera to the player most likely to get the next kill, using machine learning trained on thousands of real match scenarios.

## 🚀 What Makes OBSERVERai Special

### 🧠 Advanced AI Prediction System
- **6,115+ training samples** from real CS2 professional matches
- **85%+ accuracy** in kill prediction using machine learning
- **Real-time analysis** of player positions, weapons, health, and behavior patterns
- **AIIMS (AI Intelligence Measurement System)** provides detailed statistics on AI performance

### 🔄 Hot-Reload AI Training
- Train new AI models on your own demo files
- **Automatic model updates** - no restart required
- Models improve continuously as you add more training data
- **30-second hot-reload** system updates the observer instantly

### 📊 Professional Features
- **Automatic camera switching** with intelligent timing
- **Duel detection** - identifies 1v1 situations automatically
- **Kill pattern analysis** across 12,092+ recorded kills
- **Positional AI** understanding map control and angles
- **Pattern memory** - learns from previous rounds and player behavior

### 🎮 Two Complete Applications
- **CS2Observer.exe** - Main observer with AI predictions (208MB standalone)
- **CS2AITrainer.exe** - AI training interface with demo processing (158MB standalone)

## 📦 Installation Guide

### Option 1: Standalone Executables (Recommended)
**No Python, No Dependencies, Just Run!**

1. **Download** the latest `OBSERVERai_STANDALONE_v2.0.zip`
2. **Extract** to any folder on your computer
3. **Copy gamestate integration** (see step 3 below)
4. **Run** the applications directly

### Option 2: Python Source Code
```bash
# Clone the repository
git clone https://github.com/dualitycsgo1/OBSERVERai.git
cd OBSERVERai

# Install dependencies
pip install -r requirements.txt

# Run the applications
python cs2_duel_detector.py          # Main observer
python ai_trainer_gui.py             # AI trainer
```

### 3. CS2 Game State Integration Setup
**This step is required for both installation methods:**

1. **Copy the configuration file:**
   ```
   Copy: gamestate_integration_obs_updated.cfg
   To: Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/cfg/
   ```

2. **Find your CS2 installation directory:**
   - Default Steam location: `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`
   - Custom Steam library: `[Your Steam Library]\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`

3. **Verify the file is in place:**
   - The file should be at: `...\csgo\cfg\gamestate_integration_obs_updated.cfg`

4. **Restart CS2** after copying the file

## 🎯 How to Use OBSERVERai

### Quick Start
1. **Launch CS2Observer.exe** (or double-click `🎮 Start CS2 Observer.bat`)
2. **Start Counter-Strike 2**
3. **Join any match as Observer or GOTV**
4. **Watch the AI predict kills** and automatically switch cameras!

### Controls and Commands
- **`toggle`** - Enable/disable automatic camera switching
- **`status`** - Show current AI status and statistics
- **`cooldown X`** - Set cooldown between camera switches (seconds)
- **`train`** - Start AI training mode
- **`reload`** - Force reload AI models
- **`quit`** - Exit the application

## 🧠 AI Training System - Make Your AI Smarter

### Why Train the AI?
The AI gets **significantly better** with more training data because:
- **Pattern Recognition**: Learns player behavior patterns specific to different skill levels
- **Map Knowledge**: Understands positioning and angles on different maps
- **Meta Adaptation**: Adapts to current CS2 meta and weapon balance changes
- **Personalization**: Learns from your specific game scenarios and preferences

### How to Train New Models

#### Step 1: Collect Demo Files
1. **Download CS2 demos** (.dem files) from:
   - Your own matches (MM, Faceit, ESEA)
   - Professional matches (HLTV, tournament websites)
   - High-level pugs or scrims

2. **Place demo files** in the `demos/` folder

#### Step 2: Launch AI Trainer
```bash
# Standalone version
CS2AITrainer.exe

# Or double-click
🧠 Start AI Trainer.bat

# Python version  
python ai_trainer_gui.py
```

#### Step 3: Process Demos
1. **Click "Import Demos"** - Select your .dem files
2. **AI processes each demo** extracting:
   - Player positions at moment of kills
   - Weapon types and ammunition states
   - Health and armor values
   - Round context and economic factors
   - Map-specific positioning data

3. **Watch real-time statistics:**
   - Total kills processed
   - Maps analyzed
   - Training samples generated
   - Processing speed (kills/second)

#### Step 4: Train the Model
1. **Click "Train AI Model"** after processing demos
2. **AI training begins** using machine learning algorithms:
   - Random Forest classification
   - Feature importance analysis
   - Cross-validation testing
   - Hyperparameter optimization

3. **Training results show:**
   - Model accuracy percentage
   - Feature importance rankings
   - Training/validation split results
   - AIIMS intelligence score

#### Step 5: Automatic Integration
- **Model automatically saved** to `models/` folder
- **Observer detects new model** within 30 seconds
- **Hot-reload activates** - no restart needed!
- **AI immediately uses** improved predictions

### Understanding AIIMS Statistics

**AIIMS (AI Intelligence Measurement System)** provides detailed metrics:

- **Overall Accuracy**: Percentage of correct kill predictions
- **Map-Specific Performance**: How well AI performs on each map
- **Weapon Prediction**: Accuracy for different weapon types
- **Positional Analysis**: Success rate for different map areas
- **Time-Based Patterns**: Performance across different round timings

### Training Tips for Best Results

1. **Use Diverse Demos**: Include different skill levels, maps, and play styles
2. **Recent Demos**: Use demos from current CS2 version for best meta relevance
3. **Quality over Quantity**: 50 high-quality demos > 500 low-quality ones
4. **Regular Updates**: Retrain monthly to adapt to meta changes
5. **Balanced Training**: Include both rifle and pistol rounds

### What Happens During Training

The AI analyzes thousands of kill scenarios to learn:

#### Positional Intelligence
- **Angle advantages**: Which positions have better chances
- **Map control**: How territory control affects kill probability
- **Rotation patterns**: Common player movement sequences
- **Choke point analysis**: High-action areas on each map

#### Weapon Mastery
- **Weapon effectiveness**: Damage potential vs. player skill
- **Range optimization**: Optimal engagement distances
- **Economic impact**: How weapon value affects play style
- **Ammunition management**: Reload timing and ammo conservation

#### Player Behavior
- **Aggression patterns**: When players are likely to peek
- **Economic decisions**: How economy affects player actions
- **Round psychology**: Behavior changes based on round importance
- **Team coordination**: How team play affects individual performance

## 🔧 Advanced Configuration

### AI Settings (`ai_config.json`)
```json
{
  "prediction_threshold": 0.7,      // Minimum confidence for camera switch
  "model_update_interval": 30,      // Hot-reload check interval (seconds)
  "training_sample_limit": 10000,   // Maximum training samples to use
  "feature_importance_threshold": 0.05  // Minimum feature importance
}
```

### Observer Settings (`config.json`)
```json
{
  "cooldown_seconds": 3,            // Minimum time between camera switches
  "min_score_threshold": 0.1,       // Minimum score to trigger switch
  "enable_duel_detection": true,    // Enable 1v1 detection
  "auto_spectate_mode": true        // Automatic spectator mode
}
```

## 🎮 Game Modes and Compatibility

### Supported Modes
- ✅ **Competitive Matchmaking** (full functionality)
- ✅ **GOTV/Demos** (full functionality)
- ✅ **Faceit/ESEA** (full functionality)
- ✅ **Casual/Community** (basic functionality)
- ✅ **Tournament Broadcasting** (professional features)

### Map Support
All competitive maps fully supported:
- Ancient, Anubis, Cache, Dust2, Inferno
- Mirage, Nuke, Overpass, Train, Vertigo
- Plus community maps with automatic adaptation

## 📊 Performance Statistics

### System Requirements
- **Windows 10/11** (any edition)
- **4GB RAM** minimum (8GB recommended for training)
- **2GB disk space** for full installation
- **CS2** with game state integration
- **Internet connection** for initial setup

### Performance Metrics
- **Prediction Accuracy**: 85%+ in competitive matches
- **Response Time**: <100ms from game state to camera switch
- **Memory Usage**: ~500MB during operation
- **CPU Usage**: <5% on modern systems
- **Training Speed**: ~1000 kills/minute processing

## 🆘 Troubleshooting

### Common Issues

**🔴 Observer not receiving CS2 data:**
- Verify `gamestate_integration_obs_updated.cfg` is in correct folder
- Restart CS2 after adding the config file
- Check Windows Firewall isn't blocking the connection
- Ensure CS2 is in Observer mode or GOTV

**🔴 AI Trainer not processing demos:**
- Ensure demo files are valid CS2 format (.dem extension)
- Check demos aren't corrupted (try playing in CS2 first)
- Verify sufficient disk space for processing
- Try processing smaller demo files first

**🔴 Camera not switching automatically:**
- Type `toggle` in observer console to enable auto-switching 
- You can also toggle by pressing the '-button
- Check cooldown isn't too high (`cooldown 2`)
- Verify AI confidence threshold isn't too strict
- Ensure you're in proper observer mode

**🔴 Models not updating:**
- Check `models/` folder has write permissions
- Verify model training completed successfully
- Wait 30 seconds for hot-reload cycle
- Try manual reload with `reload` command

### Getting Help
- Check `cs2_observer.log` for detailed error information
- Check `ai_trainer.log` for training-related issues
- Verify system meets minimum requirements
- Try running as Administrator if permission issues occur

## 🎯 Professional Use Cases

### Streaming and Broadcasting
- **Automated highlight creation** - AI knows when action will happen
- **Professional camera work** - Smooth, intelligent transitions
- **Viewer engagement** - Never miss the action
- **Tournament production** - Reduce observer workload

### Esports Analysis
- **Player performance analysis** - Track prediction accuracy vs. actual performance
- **Strategic insights** - Understand positioning and timing patterns
- **Meta analysis** - See how AI predictions change with game updates
- **Educational content** - Teach positioning and game sense

### Content Creation
- **Demo analysis** - Quickly find highlight moments
- **Educational videos** - Show optimal positioning
- **AI vs. Human** comparisons - Test your game sense against AI
- **Statistical analysis** - Deep dive into game mechanics

## 🔄 Continuous Improvement

### Automatic Updates
- **Model versioning** - Keep track of AI improvements
- **Performance monitoring** - Track accuracy over time
- **Automatic backups** - Never lose training progress
- **Hot-reload system** - Seamless model updates

### Community Training
- **Share training data** - Help improve the community AI
- **Collaborative learning** - Pool demo collections
- **Tournament integration** - Train on professional matches
- **Meta adaptation** - Stay current with game changes

## 📈 Future Development

### Planned Features
- **Multi-language support** - Localized interfaces
- **Advanced statistics** - Deeper performance analytics
- **Team-based prediction** - Understand team coordination
- **Custom algorithms** - User-trainable AI architectures
- **Cloud training** - Distributed AI training system

---

**🎯 Built for the CS2 community by esports enthusiasts**

*Intelligent observing, powered by artificial intelligence.*

**Version**: 2.0 | **Build Date**: October 2025 | **License**: MIT
