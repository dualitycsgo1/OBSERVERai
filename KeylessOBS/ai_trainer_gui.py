#!/usr/bin/env python3
"""
AI Trainer GUI - User-friendly interface for training the CS2 Observer AI
Allows drag-and-drop demo training and monitoring of training progress
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
from datetime import datetime
from typing import List, Dict
import time

from ai_trainer import AITrainer, get_base_dir

class AITrainerGUI:
    """
    GUI Interface for AI Trainer
    
    Features:
    - Drag & drop demo files
    - Real-time training progress
    - Training statistics and history
    - Model performance monitoring
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CS2 Observer AI Trainer")
        self.root.geometry("800x600")
        self.root.configure(bg='#2d2d2d')
        
        # Initialize AI Trainer
        self.trainer = AITrainer()
        self.training_in_progress = False
        
        # Create GUI elements
        self.create_widgets()
        self.update_stats()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2d2d2d')
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(
            title_frame, 
            text="🧠 CS2 Observer AI Trainer",
            font=('Arial', 16, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Train your AI to become smarter by feeding it demo files",
            font=('Arial', 10),
            fg='#cccccc',
            bg='#2d2d2d'
        )
        subtitle_label.pack()
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Training Tab
        self.create_training_tab()
        
        # Statistics Tab
        self.create_statistics_tab()
        
        # History Tab
        self.create_history_tab()
        
        # AIIMS Tab (AI Intelligence Measurement System)
        self.create_aiims_tab()
        
    def create_training_tab(self):
        """Create the training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🎯 Train AI")
        
        # Demo Selection
        demo_frame = ttk.LabelFrame(training_frame, text="Demo Files", padding=10)
        demo_frame.pack(fill='x', padx=10, pady=5)
        
        # Demo list
        self.demo_listbox = tk.Listbox(
            demo_frame, 
            height=6,
            selectmode=tk.MULTIPLE,
            bg='#3d3d3d',
            fg='#ffffff',
            selectbackground='#4d4d4d'
        )
        self.demo_listbox.pack(fill='x', pady=5)
        
        # Demo buttons
        demo_buttons_frame = tk.Frame(demo_frame, bg='#f0f0f0')
        demo_buttons_frame.pack(fill='x', pady=5)
        
        ttk.Button(
            demo_buttons_frame,
            text="Add Demo Files",
            command=self.add_demo_files
        ).pack(side='left', padx=5)
        
        ttk.Button(
            demo_buttons_frame,
            text="Clear List",
            command=self.clear_demo_list
        ).pack(side='left', padx=5)
        
        ttk.Button(
            demo_buttons_frame,
            text="Scan Demos Folder",
            command=self.scan_demos_folder
        ).pack(side='left', padx=5)
        
        # Training Options
        options_frame = ttk.LabelFrame(training_frame, text="Training Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=5)
        
        # Incremental training checkbox
        self.incremental_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Incremental Training (recommended)",
            variable=self.incremental_var
        ).pack(anchor='w')
        
        # Max demos per session
        max_demos_frame = tk.Frame(options_frame)
        max_demos_frame.pack(fill='x', pady=2)
        
        tk.Label(max_demos_frame, text="Max demos per session:").pack(side='left')
        self.max_demos_var = tk.StringVar(value="10")
        ttk.Entry(
            max_demos_frame,
            textvariable=self.max_demos_var,
            width=5
        ).pack(side='left', padx=5)
        
        # Training Controls
        controls_frame = ttk.LabelFrame(training_frame, text="Training Controls", padding=10)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        self.train_button = ttk.Button(
            controls_frame,
            text="🚀 Start Training",
            command=self.start_training
        )
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(
            controls_frame,
            text="⏹ Stop Training",
            command=self.stop_training,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(training_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to train")
        self.status_label = tk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#f0f0f0'
        )
        self.status_label.pack(pady=5)
        
        # Training log
        self.log_text = scrolledtext.ScrolledText(
            progress_frame,
            height=8,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 9)
        )
        self.log_text.pack(fill='both', expand=True, pady=5)
        
    def create_statistics_tab(self):
        """Create the statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📊 Statistics")
        
        # Stats display
        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Refresh button
        ttk.Button(
            stats_frame,
            text="🔄 Refresh Statistics",
            command=self.update_stats
        ).pack(pady=5)
        
    def create_history_tab(self):
        """Create the training history tab"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="📈 History")
        
        # History display
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.history_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Refresh button
        ttk.Button(
            history_frame,
            text="🔄 Refresh History",
            command=self.update_history
        ).pack(pady=5)
        
    def create_aiims_tab(self):
        """Create the AI Intelligence Measurement System (AIIMS) tab"""
        aiims_frame = ttk.Frame(self.notebook)
        self.notebook.add(aiims_frame, text="🧠 AIIMS")
        
        # Create scrollable frame
        canvas = tk.Canvas(aiims_frame, bg='#2d2d2d')
        scrollbar = ttk.Scrollbar(aiims_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # AI Intelligence Header
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(
            header_frame,
            text="🧠 AI INTELLIGENCE MEASUREMENT SYSTEM",
            font=('Arial', 16, 'bold')
        ).pack()
        
        ttk.Label(
            header_frame,
            text="Analyze your AI's knowledge depth and prediction capabilities",
            font=('Arial', 10, 'italic')
        ).pack()
        
        # Intelligence Score Section
        self.create_intelligence_score_section(scrollable_frame)
        
        # Knowledge Depth Section
        self.create_knowledge_depth_section(scrollable_frame)
        
        # Prediction Confidence Section
        self.create_prediction_confidence_section(scrollable_frame)
        
        # Tactical Expertise Section
        self.create_tactical_expertise_section(scrollable_frame)
        
        # Learning Progress Section
        self.create_learning_progress_section(scrollable_frame)
        
        # Knowledge Gaps Section
        self.create_knowledge_gaps_section(scrollable_frame)
        
        # Map Analysis Section
        self.create_map_analysis_section(scrollable_frame)
        
        # Refresh Button
        refresh_frame = ttk.Frame(scrollable_frame)
        refresh_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(
            refresh_frame,
            text="🔄 Refresh Intelligence Analysis",
            command=self.refresh_aiims
        ).pack()
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_intelligence_score_section(self, parent):
        """Create the overall intelligence score section"""
        frame = ttk.LabelFrame(parent, text="🎯 Overall AI Intelligence Score", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        # Intelligence Score Display
        self.intelligence_score_var = tk.StringVar(value="Calculating...")
        self.intelligence_level_var = tk.StringVar(value="")
        
        score_frame = ttk.Frame(frame)
        score_frame.pack(fill='x')
        
        ttk.Label(score_frame, text="AI IQ Score:", font=('Arial', 12, 'bold')).pack(side='left')
        ttk.Label(score_frame, textvariable=self.intelligence_score_var, 
                 font=('Arial', 12, 'bold'), foreground='#00ff00').pack(side='left', padx=(10, 0))
        
        ttk.Label(frame, textvariable=self.intelligence_level_var, 
                 font=('Arial', 10, 'italic')).pack()
        
        # Score breakdown
        self.score_breakdown_text = tk.Text(frame, height=4, bg='#1e1e1e', fg='#ffffff', 
                                           font=('Consolas', 9))
        self.score_breakdown_text.pack(fill='x', pady=(5, 0))
        
    def create_knowledge_depth_section(self, parent):
        """Create the knowledge depth analysis section"""
        frame = ttk.LabelFrame(parent, text="📚 Knowledge Depth Analysis", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.knowledge_depth_text = tk.Text(frame, height=6, bg='#1e1e1e', fg='#ffffff', 
                                           font=('Consolas', 9))
        self.knowledge_depth_text.pack(fill='x')
        
    def create_prediction_confidence_section(self, parent):
        """Create the prediction confidence section"""
        frame = ttk.LabelFrame(parent, text="🔮 Prediction Confidence", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.prediction_confidence_text = tk.Text(frame, height=5, bg='#1e1e1e', fg='#ffffff', 
                                                 font=('Consolas', 9))
        self.prediction_confidence_text.pack(fill='x')
        
    def create_tactical_expertise_section(self, parent):
        """Create the tactical expertise breakdown section"""
        frame = ttk.LabelFrame(parent, text="⚔️ Tactical Expertise Breakdown", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.tactical_expertise_text = tk.Text(frame, height=8, bg='#1e1e1e', fg='#ffffff', 
                                              font=('Consolas', 9))
        self.tactical_expertise_text.pack(fill='x')
        
    def create_learning_progress_section(self, parent):
        """Create the learning progress section"""
        frame = ttk.LabelFrame(parent, text="📈 Learning Progress Over Time", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.learning_progress_text = tk.Text(frame, height=6, bg='#1e1e1e', fg='#ffffff', 
                                             font=('Consolas', 9))
        self.learning_progress_text.pack(fill='x')
        
    def create_knowledge_gaps_section(self, parent):
        """Create the knowledge gaps section"""
        frame = ttk.LabelFrame(parent, text="🎯 Knowledge Gaps & Recommendations", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.knowledge_gaps_text = tk.Text(frame, height=6, bg='#1e1e1e', fg='#ffffff', 
                                          font=('Consolas', 9))
        self.knowledge_gaps_text.pack(fill='x')
    
    def create_map_analysis_section(self, parent):
        """Create the per-map training analysis section"""
        frame = ttk.LabelFrame(parent, text="🗺️ Map Distribution Analysis", padding=10)
        frame.pack(fill='x', padx=10, pady=5)
        
        self.map_analysis_text = tk.Text(frame, height=8, bg='#1e1e1e', fg='#ffffff', 
                                        font=('Consolas', 9))
        self.map_analysis_text.pack(fill='x')
        
    def add_demo_files(self):
        """Add demo files via file dialog"""
        files = filedialog.askopenfilenames(
            title="Select Demo Files",
            filetypes=[("Demo files", "*.dem"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.demo_listbox.get(0, tk.END):
                self.demo_listbox.insert(tk.END, file)
                
    def clear_demo_list(self):
        """Clear the demo list"""
        self.demo_listbox.delete(0, tk.END)
        
    def scan_demos_folder(self):
        """Scan the demos folder for new files"""
        new_demos = self.trainer._find_new_demos()
        
        self.demo_listbox.delete(0, tk.END)
        for demo in new_demos:
            self.demo_listbox.insert(tk.END, demo)
            
        self.log(f"Found {len(new_demos)} new demos in folder")
        
    def start_training(self):
        """Start the training process"""
        if self.training_in_progress:
            return
            
        # Get selected demos
        demo_files = list(self.demo_listbox.get(0, tk.END))
        
        if not demo_files:
            messagebox.showwarning("No Demos", "Please add demo files first")
            return
            
        # Convert relative paths to absolute paths
        absolute_demo_files = []
        for demo_file in demo_files:
            if not os.path.isabs(demo_file):
                absolute_path = os.path.abspath(demo_file)
                absolute_demo_files.append(absolute_path)
                self.log(f"Debug: Converting {demo_file} to {absolute_path}")
            else:
                absolute_demo_files.append(demo_file)
                self.log(f"Debug: Using absolute path {demo_file}")
        
        demo_files = absolute_demo_files
            
        # Update UI
        self.training_in_progress = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_var.set(0)
        self.status_var.set("Starting training...")
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self.training_worker,
            args=(demo_files,),
            daemon=True
        )
        self.training_thread.start()
        
    def stop_training(self):
        """Stop the training process"""
        self.training_in_progress = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Training stopped by user")
        
    def training_worker(self, demo_files: List[str]):
        """Training worker thread"""
        try:
            self.log("🚀 Starting AI training session...")
            self.log(f"Processing {len(demo_files)} demo files")
            
            # Debug: Log the demo files being processed
            for i, demo_file in enumerate(demo_files):
                self.log(f"Demo {i+1}: {demo_file}")
                self.log(f"  Exists: {os.path.exists(demo_file)}")
                if os.path.exists(demo_file):
                    self.log(f"  Size: {os.path.getsize(demo_file)} bytes")
            
            # Step 1: Process demos
            self.root.after(0, lambda: self.status_var.set("Processing demo files..."))
            self.root.after(0, lambda: self.progress_var.set(20))
            
            results = self.trainer.process_new_demos(demo_files)
            
            self.log(f"✅ Processed {results['processed']} demos")
            self.log(f"   - New kills: {results['new_kills']}")
            self.log(f"   - Failed: {results['failed']}")
            
            # Debug: Log any errors
            if results.get('errors'):
                self.log("❌ Errors encountered:")
                for error in results['errors']:
                    self.log(f"   {error}")
            
            if results['processed'] == 0:
                self.log("❌ No demos were successfully processed")
                self.training_finished()
                return
                
            # Step 2: Train model
            self.root.after(0, lambda: self.status_var.set("Training AI model..."))
            self.root.after(0, lambda: self.progress_var.set(60))
            
            incremental = self.incremental_var.get()
            training_results = self.trainer.train_model(incremental=incremental)
            
            if training_results.get('success'):
                accuracy = training_results.get('accuracy', 0)
                self.log(f"✅ Model training completed!")
                self.log(f"   - Accuracy: {accuracy:.3f}")
                self.log(f"   - Training samples: {training_results.get('training_samples', 0)}")
            else:
                self.log(f"❌ Model training failed: {training_results.get('error', 'Unknown error')}")
                
            # Step 3: Integrate model
            self.root.after(0, lambda: self.status_var.set("Integrating model..."))
            self.root.after(0, lambda: self.progress_var.set(80))
            
            integration_success = self.trainer.auto_integrate_model()
            
            if integration_success:
                self.log("✅ Model integrated successfully")
                self.log("🎯 AI system will use the updated model")
            else:
                self.log("⚠️  Model integration had issues")
                
            # Complete
            self.root.after(0, lambda: self.progress_var.set(100))
            self.log("🎉 Training session completed!")
            
        except Exception as e:
            self.log(f"❌ Training failed: {str(e)}")
            
        finally:
            self.training_finished()
            
    def training_finished(self):
        """Called when training is finished"""
        self.training_in_progress = False
        self.root.after(0, lambda: self.train_button.config(state='normal'))
        self.root.after(0, lambda: self.stop_button.config(state='disabled'))
        self.root.after(0, lambda: self.status_var.set("Training completed"))
        
        # Update statistics
        self.root.after(1000, self.update_stats)
        
    def log(self, message: str):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
            
        self.root.after(0, update_log)
        
    def update_stats(self):
        """Update statistics display"""
        try:
            stats = self.trainer.get_training_stats()
            
            stats_text = "📊 COMPREHENSIVE AI TRAINING STATISTICS\n"
            stats_text += "=" * 55 + "\n\n"
            
            # Current Training Session Stats
            stats_text += "🔄 CURRENT TRAINING SESSIONS:\n"
            stats_text += f"   🎯 Sessions: {stats['total_sessions']}\n"
            stats_text += f"   🎮 Demos: {stats['current_demos_processed']}\n"
            stats_text += f"   💀 Kills: {stats['current_kills_analyzed']}\n\n"
            
            # Historical Dataset Stats
            stats_text += "📚 HISTORICAL DATASET (85 Demos):\n"
            stats_text += f"   📁 Demos: {stats['historical_demos']}\n"
            stats_text += f"   💀 Kills: {stats['historical_kills']:,}\n"
            stats_text += f"   🎯 Training Samples: {stats['historical_training_samples']:,}\n\n"
            
            # Combined Totals
            stats_text += "🏆 TOTAL STATISTICS:\n"
            stats_text += f"   🎮 Total Demos: {stats['total_demos_processed']}\n"
            stats_text += f"   💀 Total Kills: {stats['total_kills_analyzed']:,}\n"
            stats_text += f"   🎯 Total Training Samples: {stats['total_training_samples']:,}\n\n"
            
            # Model Performance
            stats_text += "🤖 MODEL PERFORMANCE:\n"
            stats_text += f"   🧠 Model Versions: {stats['model_versions']}\n"
            stats_text += f"   🎯 Latest Accuracy: {stats['latest_accuracy']:.3f} ({stats['latest_accuracy']*100:.1f}%)\n\n"
            
            stats_text += f"📁 Data Files: {len(stats['training_data_files'])}\n"
            stats_text += f"🤖 Model Files: {len(stats['model_files'])}\n\n"
            
            if stats['training_data_files']:
                stats_text += "📋 Training Data Files:\n"
                for file in stats['training_data_files'][-5:]:  # Show last 5
                    stats_text += f"   - {file}\n"
                    
            if stats['model_files']:
                stats_text += "\n🤖 Model Files:\n"
                for file in stats['model_files']:
                    stats_text += f"   - {file}\n"
                    
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
            # Also refresh AIIMS data
            try:
                self.refresh_aiims()
            except:
                pass  # AIIMS refresh is optional
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error loading statistics: {e}")
            
    def update_history(self):
        """Update training history display"""
        try:
            history_text = "📈 TRAINING HISTORY\n"
            history_text += "=" * 50 + "\n\n"
            
            history = self.trainer.training_history
            sessions = history.get('sessions', [])
            
            if not sessions:
                history_text += "No training sessions yet.\n"
            else:
                for i, session in enumerate(sessions[-10:], 1):  # Show last 10
                    timestamp = session.get('timestamp', 'Unknown')
                    results = session.get('training_results', {})
                    
                    history_text += f"Session {i} - {timestamp}\n"
                    history_text += f"   Accuracy: {results.get('accuracy', 0):.3f}\n"
                    history_text += f"   Samples: {results.get('training_samples', 0)}\n"
                    history_text += f"   Success: {'✅' if results.get('success') else '❌'}\n\n"
                    
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(1.0, history_text)
            
        except Exception as e:
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(1.0, f"Error loading history: {e}")
            
    def refresh_aiims(self):
        """Refresh the AIIMS analysis"""
        try:
            # Calculate AI intelligence metrics
            intelligence_data = self.calculate_ai_intelligence()
            
            # Update displays
            self.update_intelligence_score(intelligence_data)
            self.update_knowledge_depth(intelligence_data)
            self.update_prediction_confidence(intelligence_data)
            self.update_tactical_expertise(intelligence_data)
            self.update_learning_progress(intelligence_data)
            self.update_knowledge_gaps(intelligence_data)
            self.update_map_analysis(intelligence_data)
            
        except Exception as e:
            self.log(f"Error refreshing AIIMS: {e}")
    
    def calculate_ai_intelligence(self):
        """Calculate comprehensive AI intelligence metrics"""
        try:
            import pickle
            import os
            from sklearn.ensemble import RandomForestClassifier
            
            # Get statistics with error handling
            try:
                stats = self.trainer.get_training_stats()
                self.log(f"AIIMS: Got stats: {list(stats.keys()) if stats else 'None'}")
            except Exception as e:
                self.log(f"AIIMS: Error getting stats: {e}")
                stats = {}
            
            # Base intelligence factors
            data_factor = min(stats.get('total_training_samples', 0) / 10000, 1.0)  # Max at 10k samples
            demo_factor = min(stats.get('total_demos_processed', 0) / 100, 1.0)    # Max at 100 demos
            accuracy_factor = stats.get('latest_accuracy', 0)
            
            self.log(f"AIIMS: Factors - data:{data_factor:.3f}, demo:{demo_factor:.3f}, accuracy:{accuracy_factor:.3f}")
            
            # Model complexity factor
            model_complexity = 0.5  # Default
            model_path = os.path.join(self.trainer.models_dir, "observer_ai_model.pkl")
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    if hasattr(model, 'n_estimators'):
                        model_complexity = min(model.n_estimators / 200, 1.0)  # Max at 200 trees
                    self.log(f"AIIMS: Model loaded, complexity: {model_complexity:.3f}")
                except Exception as e:
                    self.log(f"AIIMS: Error loading model: {e}")
            else:
                self.log(f"AIIMS: Model not found at {model_path}")
            
            # Experience diversity factor (based on different demo types)
            experience_factor = min(len(stats.get('training_data_files', [])) / 20, 1.0)
            
            # Calculate overall intelligence score (0-100)
            intelligence_score = (
                data_factor * 30 +        # 30% weight on training data volume
                demo_factor * 25 +        # 25% weight on demo variety
                accuracy_factor * 25 +    # 25% weight on prediction accuracy
                model_complexity * 10 +   # 10% weight on model complexity
                experience_factor * 10    # 10% weight on experience diversity
            )
            
            # Determine intelligence level
            if intelligence_score >= 90:
                level = "🏆 GRANDMASTER - Elite tactical AI"
            elif intelligence_score >= 80:
                level = "💎 EXPERT - Advanced predictive capabilities"
            elif intelligence_score >= 70:
                level = "🥇 ADVANCED - Strong tactical understanding"
            elif intelligence_score >= 60:
                level = "🥈 INTERMEDIATE - Good pattern recognition"
            elif intelligence_score >= 40:
                level = "🥉 BASIC - Learning fundamentals"
            else:
                level = "🌱 NOVICE - Early learning stage"
            
            return {
                'intelligence_score': intelligence_score,
                'intelligence_level': level,
                'data_factor': data_factor,
                'demo_factor': demo_factor,
                'accuracy_factor': accuracy_factor,
                'model_complexity': model_complexity,
                'experience_factor': experience_factor,
                'stats': stats,
                'model_exists': os.path.exists(model_path)
            }
            
        except Exception as e:
            self.log(f"AIIMS: Critical error in calculate_ai_intelligence: {e}")
            import traceback
            self.log(f"AIIMS: Traceback: {traceback.format_exc()}")
            return {
                'intelligence_score': 0,
                'intelligence_level': f'❌ ERROR - {str(e)[:50]}...' if len(str(e)) > 50 else f'❌ ERROR - {str(e)}',
                'error': str(e),
                'stats': {}
            }
    
    def update_intelligence_score(self, data):
        """Update the intelligence score display"""
        score = data.get('intelligence_score', 0)
        level = data.get('intelligence_level', 'Unknown')
        
        self.intelligence_score_var.set(f"{score:.1f}/100")
        self.intelligence_level_var.set(level)
        
        # Score breakdown
        breakdown = f"""
Data Volume:     {data.get('data_factor', 0)*100:5.1f}% (Training samples: {data.get('stats', {}).get('total_training_samples', 0):,})
Demo Variety:    {data.get('demo_factor', 0)*100:5.1f}% (Demos processed: {data.get('stats', {}).get('total_demos_processed', 0):,})
Accuracy:        {data.get('accuracy_factor', 0)*100:5.1f}% (Latest model accuracy)
Model Complex:   {data.get('model_complexity', 0)*100:5.1f}% (Algorithm sophistication)
        """.strip()
        
        self.score_breakdown_text.delete(1.0, tk.END)
        self.score_breakdown_text.insert(1.0, breakdown)
    
    def update_knowledge_depth(self, data):
        """Update the knowledge depth analysis"""
        stats = data.get('stats', {})
        
        analysis = f"""
📊 TRAINING DATA ANALYSIS:
• Total Training Samples: {stats.get('total_training_samples', 0):,}
• Historical Dataset: {stats.get('historical_training_samples', 0):,} samples
• Current Sessions: {stats.get('current_training_samples', 0):,} samples
• Demos Analyzed: {stats.get('total_demos_processed', 0)} total
• Kill Situations: {stats.get('total_kills_analyzed', 0):,} analyzed

🎯 DATA QUALITY:
• Model Versions: {stats.get('model_versions', 0)} iterations
• Training Files: {len(stats.get('training_data_files', []))} batches
• Model Files: {len(stats.get('model_files', []))} variants
        """.strip()
        
        self.knowledge_depth_text.delete(1.0, tk.END)
        self.knowledge_depth_text.insert(1.0, analysis)
    
    def update_prediction_confidence(self, data):
        """Update the prediction confidence analysis"""
        stats = data.get('stats', {})
        accuracy = stats.get('latest_accuracy', 0)
        
        if accuracy > 0.9:
            confidence_level = "🎯 EXTREMELY HIGH - Predictions very reliable"
        elif accuracy > 0.8:
            confidence_level = "✅ HIGH - Good prediction reliability"
        elif accuracy > 0.7:
            confidence_level = "⚠️ MODERATE - Decent prediction accuracy"
        elif accuracy > 0.5:
            confidence_level = "❌ LOW - Predictions unreliable"
        else:
            confidence_level = "🚫 VERY LOW - AI needs more training"
        
        analysis = f"""
🔮 PREDICTION CONFIDENCE: {confidence_level}
• Model Accuracy: {accuracy*100:.1f}%
• Confidence Level: {min(accuracy * 1.2, 1.0)*100:.1f}%
• Reliability Score: {'🌟' * int(accuracy * 5)}

💡 PREDICTION CAPABILITIES:
• Duel Outcomes: {'Excellent' if accuracy > 0.85 else 'Good' if accuracy > 0.7 else 'Needs Improvement'}
• Player Selection: {'High Confidence' if data.get('model_exists') else 'Model Not Found'}
• Tactical Analysis: {'Advanced' if stats.get('total_training_samples', 0) > 1000 else 'Basic'}
        """.strip()
        
        self.prediction_confidence_text.delete(1.0, tk.END)
        self.prediction_confidence_text.insert(1.0, analysis)
    
    def update_tactical_expertise(self, data):
        """Update the tactical expertise breakdown"""
        stats = data.get('stats', {})
        samples = stats.get('total_training_samples', 0)
        
        # Estimate expertise levels based on training data
        def get_expertise_level(sample_count, threshold_expert=2000, threshold_good=500):
            if sample_count >= threshold_expert:
                return "🏆 EXPERT"
            elif sample_count >= threshold_good:
                return "🥇 GOOD"
            elif sample_count >= 100:
                return "🥈 BASIC"
            else:
                return "🥉 LEARNING"
        
        expertise = get_expertise_level(samples)
        
        analysis = f"""
⚔️ TACTICAL EXPERTISE: {expertise}

🗺️ MAP KNOWLEDGE:
• Map Understanding: {get_expertise_level(samples, 1500, 300)}
• Position Analysis: {get_expertise_level(samples, 2000, 500)}
• Line of Sight: {get_expertise_level(samples, 1000, 200)}

🔫 WEAPON PROFICIENCY:
• Damage Prediction: {get_expertise_level(samples, 1800, 400)}
• Weapon Selection: {get_expertise_level(samples, 1200, 250)}
• Range Analysis: {get_expertise_level(samples, 1500, 350)}

🎯 SITUATION HANDLING:
• Clutch Scenarios: {get_expertise_level(samples, 2500, 600)}
• Trade Kills: {get_expertise_level(samples, 2000, 400)}
• Entry Frags: {get_expertise_level(samples, 1800, 350)}
        """.strip()
        
        self.tactical_expertise_text.delete(1.0, tk.END)
        self.tactical_expertise_text.insert(1.0, analysis)
    
    def update_learning_progress(self, data):
        """Update the learning progress analysis"""
        stats = data.get('stats', {})
        
        # Analyze learning progression
        sessions = len(stats.get('training_data_files', []))
        total_samples = stats.get('total_training_samples', 0)
        accuracy = stats.get('latest_accuracy', 0)
        
        progress_analysis = f"""
📈 LEARNING PROGRESSION:
• Training Sessions: {sessions}
• Total Learning Events: {total_samples:,}
• Current Performance: {accuracy*100:.1f}%
• Learning Rate: {'Excellent' if sessions > 10 else 'Good' if sessions > 5 else 'Getting Started'}

🎓 MILESTONES ACHIEVED:
{'✅' if total_samples >= 1000 else '⏳'} 1,000 Training Samples {'(ACHIEVED)' if total_samples >= 1000 else f'({total_samples}/1,000)'}
{'✅' if total_samples >= 5000 else '⏳'} 5,000 Training Samples {'(ACHIEVED)' if total_samples >= 5000 else f'({total_samples}/5,000)'}
{'✅' if accuracy >= 0.8 else '⏳'} 80% Accuracy {'(ACHIEVED)' if accuracy >= 0.8 else f'({accuracy*100:.1f}%/80%)'}
{'✅' if sessions >= 10 else '⏳'} 10 Training Sessions {'(ACHIEVED)' if sessions >= 10 else f'({sessions}/10)'}

🚀 NEXT TARGETS:
• {'Reach 10,000 samples' if total_samples < 10000 else 'Maintain excellence'}
• {'Achieve 90% accuracy' if accuracy < 0.9 else 'Perfect prediction model'}
        """.strip()
        
        self.learning_progress_text.delete(1.0, tk.END)
        self.learning_progress_text.insert(1.0, progress_analysis)
    
    def update_knowledge_gaps(self, data):
        """Update the knowledge gaps and recommendations"""
        stats = data.get('stats', {})
        samples = stats.get('total_training_samples', 0)
        accuracy = stats.get('latest_accuracy', 0)
        demos = stats.get('total_demos_processed', 0)
        
        gaps = []
        recommendations = []
        
        if samples < 1000:
            gaps.append("📊 Limited training data")
            recommendations.append("• Feed more demo files to increase knowledge base")
        
        if accuracy < 0.8:
            gaps.append("🎯 Low prediction accuracy")
            recommendations.append("• Focus on high-quality professional demos")
        
        if demos < 20:
            gaps.append("🎮 Limited demo variety")
            recommendations.append("• Add demos from different maps and skill levels")
        
        if len(stats.get('training_data_files', [])) < 5:
            gaps.append("📈 Few training sessions")
            recommendations.append("• Run more training sessions to improve model")
        
        if not gaps:
            gaps.append("🎉 No major gaps detected!")
            recommendations.append("• Continue feeding quality demos")
            recommendations.append("• Monitor performance and fine-tune")
        
        analysis = f"""
🎯 IDENTIFIED KNOWLEDGE GAPS:
{chr(10).join(gaps)}

💡 RECOMMENDATIONS:
{chr(10).join(recommendations)}

🚀 PRIORITY ACTIONS:
{'1. Add more training data (Priority: HIGH)' if samples < 1000 else '1. Maintain data quality (Priority: MEDIUM)'}
{'2. Improve model accuracy (Priority: HIGH)' if accuracy < 0.8 else '2. Fine-tune existing model (Priority: LOW)'}
3. Expand demo variety (Priority: MEDIUM)
        """.strip()
        
        self.knowledge_gaps_text.delete(1.0, tk.END)
        self.knowledge_gaps_text.insert(1.0, analysis)
    
    def analyze_map_distribution(self):
        """Analyze the distribution of training data across different maps/demos"""
        try:
            # Get demo file distribution from current training data
            demo_stats = {}
            total_samples = 0
            
            # Analyze current training data files
            training_data_dir = self.trainer.training_data_dir
            if os.path.exists(training_data_dir):
                try:
                    filenames = os.listdir(training_data_dir)
                except (OSError, UnicodeDecodeError) as e:
                    self.log(f"Error reading training data directory: {e}")
                    filenames = []
                    
                for filename in filenames:
                    if filename.endswith('.json') and filename.startswith('training_batch_'):
                        filepath = os.path.join(training_data_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            for sample in data:
                                demo_file = sample.get('demo_file', 'unknown')
                                # Extract potential map name from demo filename
                                map_name = self.extract_map_from_demo_name(demo_file)
                                
                                if map_name not in demo_stats:
                                    demo_stats[map_name] = {
                                        'samples': 0,
                                        'demo_files': set(),
                                        'weapons': set(),
                                        'rounds': set()
                                    }
                                
                                demo_stats[map_name]['samples'] += 1
                                demo_stats[map_name]['demo_files'].add(demo_file)
                                demo_stats[map_name]['weapons'].add(sample.get('weapon', 'unknown'))
                                demo_stats[map_name]['rounds'].add(sample.get('round_num', 0))
                                total_samples += 1
                                
                        except Exception as e:
                            self.log(f"Error processing {filename}: {e}")
                            continue
            
            # Analyze historical dataset if available
            try:
                historical_path = os.path.join(get_base_dir(), "complete_positional_dataset_all_85_demos.json")
                historical_path = os.path.normpath(historical_path)  # Normalize path separators
            except Exception as e:
                self.log(f"Error constructing historical dataset path: {e}")
                historical_path = None
                
            if historical_path and os.path.exists(historical_path):
                try:
                    with open(historical_path, 'r', encoding='utf-8') as f:
                        historical_data = json.load(f)
                    
                    # Count demos in historical dataset
                    if 'training_samples' in historical_data and isinstance(historical_data['training_samples'], list):
                        samples_processed = 0
                        for sample in historical_data['training_samples']:
                            if not isinstance(sample, dict):
                                continue
                                
                            demo_file = sample.get('demo_file', 'unknown')
                            map_name = self.extract_map_from_demo_name(demo_file)
                            
                            if map_name not in demo_stats:
                                demo_stats[map_name] = {
                                    'samples': 0,
                                    'demo_files': set(),
                                    'weapons': set(),
                                    'rounds': set()
                                }
                            
                            demo_stats[map_name]['samples'] += 1
                            demo_stats[map_name]['demo_files'].add(demo_file)
                            total_samples += 1
                            samples_processed += 1
                            
                            # Prevent memory issues with very large datasets
                            if samples_processed > 50000:
                                self.log("Large dataset detected, processing first 50,000 samples for map analysis")
                                break
                            
                except Exception as e:
                    self.log(f"Error processing historical data: {e}")
            
            return demo_stats, total_samples
            
        except Exception as e:
            self.log(f"Error analyzing map distribution: {e}")
            return {}, 0
    
    def extract_map_from_demo_name(self, demo_file, demo_path=None):
        """Extract potential map name from demo filename or file"""
        if not demo_file or demo_file == 'unknown':
            return 'Unknown/Mixed'
        
        demo_lower = demo_file.lower()
        
        # Common CS2 map names with de_ prefix support
        map_names = {
            'dust2': 'Dust2',
            'mirage': 'Mirage', 
            'inferno': 'Inferno',
            'cache': 'Cache',
            'overpass': 'Overpass',
            'train': 'Train',
            'nuke': 'Nuke',
            'vertigo': 'Vertigo',
            'ancient': 'Ancient',
            'anubis': 'Anubis',
            'tuscan': 'Tuscan',
            'cobblestone': 'Cobblestone',
            'cbble': 'Cobblestone',
            'canals': 'Canals',
            'austria': 'Austria',
            'biome': 'Biome',
            'subzero': 'Subzero',
            'abbey': 'Abbey',
            'agency': 'Agency',
            'office': 'Office'
        }
        
        # Check for map names in filename (including de_ prefix)
        for map_key, map_display in map_names.items():
            if map_key in demo_lower or f'de_{map_key}' in demo_lower:
                return map_display
        
        # Try to extract map from actual demo file if path is provided
        if demo_path and os.path.exists(demo_path):
            try:
                from comprehensive_demo_processor import ComprehensiveDemoProcessor
                processor = ComprehensiveDemoProcessor()
                header_info = processor.extract_header_info(demo_path)
                
                if header_info and header_info.get('success') and header_info.get('map_name'):
                    extracted_map = header_info['map_name']
                    # Clean up the map name (remove de_ prefix and capitalize)
                    if extracted_map.startswith('de_'):
                        extracted_map = extracted_map[3:]  # Remove 'de_' prefix
                    
                    # Map to display names
                    for map_key, map_display in map_names.items():
                        if map_key.lower() == extracted_map.lower():
                            return map_display
                    return extracted_map.capitalize()
            except Exception as e:
                self.log(f"Error extracting map from demo file: {e}")
        
        # Don't show GUID-based demo files as individual maps
        # Instead group them as generic demos
        if demo_file.startswith('1-') and len(demo_file) > 30:
            return 'Custom/Practice Demos'
        
        return 'Unknown/Mixed'
    
    def update_map_analysis(self, data):
        """Update the map analysis display"""
        demo_stats, total_samples = self.analyze_map_distribution()
        
        if not demo_stats:
            analysis = """
🗺️ MAP ANALYSIS: No data available
Unable to analyze map distribution from current training data.
Add more demos to see per-map statistics.
            """.strip()
        else:
            # Sort by number of samples (descending)
            sorted_maps = sorted(demo_stats.items(), key=lambda x: x[1]['samples'], reverse=True)
            
            analysis = f"🗺️ MAP/DEMO DISTRIBUTION ANALYSIS:\n"
            analysis += f"Total Training Samples: {total_samples:,}\n\n"
            
            # Show top maps/demos
            analysis += "📊 SAMPLE DISTRIBUTION:\n"
            for i, (map_name, stats) in enumerate(sorted_maps[:10], 1):
                samples = stats['samples']
                percentage = (samples / total_samples * 100) if total_samples > 0 else 0
                demo_files = len(stats['demo_files'])
                
                # Determine if this map needs more data
                status = ""
                if samples < 100:
                    status = " ⚠️ NEEDS MORE"
                elif samples < 300:
                    status = " 📈 GROWING"
                elif samples >= 500:
                    status = " ✅ STRONG"
                
                analysis += f"{i:2d}. {map_name:<20} {samples:>4,} samples ({percentage:5.1f}%) - {demo_files} demos{status}\n"
            
            if len(sorted_maps) > 10:
                analysis += f"    ... and {len(sorted_maps) - 10} more maps/demos\n"
            
            # Recommendations
            analysis += "\n💡 TRAINING RECOMMENDATIONS:\n"
            
            weak_maps = [name for name, stats in sorted_maps if stats['samples'] < 100]
            if weak_maps:
                analysis += f"• Need more demos for: {', '.join(weak_maps[:5])}\n"
            
            if len(sorted_maps) < 5:
                analysis += "• Add demos from more diverse maps\n"
                
            strong_maps = [name for name, stats in sorted_maps if stats['samples'] >= 500]
            if strong_maps:
                analysis += f"• Well-trained on: {', '.join(strong_maps[:3])}\n"
        
        self.map_analysis_text.delete(1.0, tk.END)
        self.map_analysis_text.insert(1.0, analysis)
            
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

class AITrainerCLI:
    """
    Command-line interface for AI Trainer
    """
    
    def __init__(self):
        self.trainer = AITrainer()
        
    def run_interactive(self):
        """Run interactive CLI mode"""
        print("🧠 CS2 Observer AI Trainer - CLI Mode")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. 📊 Show statistics")
            print("2. 🎯 Train with new demos")
            print("3. 📁 Scan for new demos")
            print("4. 📈 Show training history")
            print("5. 🔄 Auto-train (process + train)")
            print("6. ❌ Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.show_statistics()
            elif choice == '2':
                self.train_with_demos()
            elif choice == '3':
                self.scan_demos()
            elif choice == '4':
                self.show_history()
            elif choice == '5':
                self.auto_train()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
                
    def show_statistics(self):
        """Show comprehensive training statistics"""
        stats = self.trainer.get_training_stats()
        
        print("\n📊 COMPREHENSIVE AI TRAINING STATISTICS")
        print("=" * 55)
        
        # Current Training Session Stats
        print("\n🔄 CURRENT TRAINING SESSIONS:")
        print(f"   🎯 Sessions: {stats['total_sessions']}")
        print(f"   🎮 Demos: {stats['current_demos_processed']}")
        print(f"   💀 Kills: {stats['current_kills_analyzed']}")
        
        # Historical Dataset Stats  
        print("\n📚 HISTORICAL DATASET (85 Demos):")
        print(f"   📁 Demos: {stats['historical_demos']}")
        print(f"   💀 Kills: {stats['historical_kills']:,}")
        print(f"   🎯 Training Samples: {stats['historical_training_samples']:,}")
        
        # Combined Totals
        print("\n🏆 TOTAL STATISTICS:")
        print(f"   🎮 Total Demos: {stats['total_demos_processed']}")
        print(f"   💀 Total Kills: {stats['total_kills_analyzed']:,}")
        print(f"   🎯 Total Training Samples: {stats['total_training_samples']:,}")
        
        # Model Performance
        print("\n🤖 MODEL PERFORMANCE:")
        print(f"   🧠 Model Versions: {stats['model_versions']}")
        print(f"   🎯 Latest Accuracy: {stats['latest_accuracy']:.3f} ({stats['latest_accuracy']*100:.1f}%)")
        
        # File Information
        print("\n📋 FILES:")
        print(f"   📁 Data Files: {len(stats['training_data_files'])}")
        print(f"   🤖 Model Files: {len(stats['model_files'])}")
        print()
        
    def scan_demos(self):
        """Scan for new demos"""
        print("\n🔍 Scanning for new demos...")
        new_demos = self.trainer._find_new_demos()
        
        if new_demos:
            print(f"Found {len(new_demos)} new demos:")
            for demo in new_demos:
                print(f"   - {os.path.basename(demo)}")
        else:
            print("No new demos found")
            
    def auto_train(self):
        """Automatically process and train"""
        print("\n🚀 Starting auto-training...")
        
        # Process new demos
        print("📁 Processing new demos...")
        results = self.trainer.process_new_demos()
        
        print(f"✅ Processed {results['processed']} demos")
        print(f"   - New kills: {results['new_kills']}")
        print(f"   - Failed: {results['failed']}")
        
        if results['processed'] > 0:
            # Train model
            print("🧠 Training model...")
            training_results = self.trainer.train_model(incremental=True)
            
            if training_results.get('success'):
                print(f"✅ Training completed!")
                print(f"   - Accuracy: {training_results.get('accuracy', 0):.3f}")
                
                # Integrate
                if self.trainer.auto_integrate_model():
                    print("✅ Model integrated successfully")
                else:
                    print("⚠️  Model integration issues")
            else:
                print(f"❌ Training failed: {training_results.get('error')}")
        else:
            print("❌ No new data to train on")
            
    def train_with_demos(self):
        """Train with specific demo files"""
        demo_files = []
        
        print("\nEnter demo file paths (empty line to finish):")
        while True:
            path = input("Demo file: ").strip()
            if not path:
                break
            if os.path.exists(path):
                demo_files.append(path)
            else:
                print(f"❌ File not found: {path}")
                
        if demo_files:
            print(f"\n🎯 Training with {len(demo_files)} demos...")
            results = self.trainer.process_new_demos(demo_files)
            
            print(f"✅ Processing results:")
            print(f"   - Processed: {results['processed']}")
            print(f"   - New kills: {results['new_kills']}")
            print(f"   - Failed: {results['failed']}")
            
            if results['processed'] > 0:
                training_results = self.trainer.train_model(incremental=True)
                if training_results.get('success'):
                    print(f"✅ Model trained: {training_results.get('accuracy', 0):.3f} accuracy")
                else:
                    print(f"❌ Training failed")
        else:
            print("❌ No valid demo files provided")
            
    def show_history(self):
        """Show training history"""
        history = self.trainer.training_history
        sessions = history.get('sessions', [])
        
        print("\n📈 TRAINING HISTORY")
        print("=" * 50)
        
        if not sessions:
            print("No training sessions yet.")
            return
            
        for i, session in enumerate(sessions[-5:], 1):  # Show last 5
            timestamp = session.get('timestamp', 'Unknown')
            results = session.get('training_results', {})
            
            print(f"\nSession {i} - {timestamp}")
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
            print(f"   Samples: {results.get('training_samples', 0)}")
            print(f"   Success: {'✅' if results.get('success') else '❌'}")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # CLI mode
        cli = AITrainerCLI()
        cli.run_interactive()
    else:
        # GUI mode
        try:
            gui = AITrainerGUI()
            gui.run()
        except Exception as e:
            print(f"GUI failed to start: {e}")
            print("Falling back to CLI mode...")
            cli = AITrainerCLI()
            cli.run_interactive()

if __name__ == "__main__":
    main()