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
            text="Evaluate future-kill models on demos the model has not trained on",
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
            text="Retrain on accumulated data (recommended)",
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
            text="MODEL EVALUATION",
            font=('Arial', 16, 'bold')
        ).pack()

        ttk.Label(
            header_frame,
            text="Measure prediction quality on independent demo files",
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
        frame = ttk.LabelFrame(parent, text="Held-out evaluation", padding=10)
        frame.pack(fill='x', padx=10, pady=5)

        # Intelligence Score Display
        self.intelligence_score_var = tk.StringVar(value="Calculating...")
        self.intelligence_level_var = tk.StringVar(value="")

        score_frame = ttk.Frame(frame)
        score_frame.pack(fill='x')

        ttk.Label(score_frame, text="Evaluation:", font=('Arial', 12, 'bold')).pack(side='left')
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
            filetypes=[("Demo files", "*.dem *.dem.zst"), ("All files", "*.*")]
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

            if not training_results.get('success'):
                self.log(f"Training failed: {training_results.get('error', 'Unknown error')}")
                return
            evaluation = training_results.get('evaluation', {})
            candidate = evaluation.get('candidate', {})
            self.log("Candidate evaluated on held-out demos (two-second future-kill target)")
            self.log(f"   - Brier score: {candidate.get('brier_score', 0):.4f} (lower is better)")
            self.log(f"   - Average precision: {candidate.get('average_precision', 0):.4f}")
            self.log(f"   - Top-player kill coverage: {candidate.get('top1_kill_coverage', 0):.3f}")
            self.log(f"   - Training/held-out samples: {training_results.get('training_samples', 0)}/{training_results.get('test_samples', 0)}")
            if not training_results.get('published'):
                self.log(training_results.get('message', 'Candidate did not pass deployment evaluation'))
                self.root.after(0, lambda: self.status_var.set("Evaluated; deployed model unchanged"))
                return
            self.root.after(0, lambda: self.status_var.set("Integrating evaluated model..."))
            self.root.after(0, lambda: self.progress_var.set(80))
            if self.trainer.auto_integrate_model():
                self.log("Evaluated model published; observer can hot-reload it")
            else:
                self.log("Published model could not be verified for integration")

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
            stats_text += "📚 LEGACY DATA (excluded from prediction training):\n"
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
        """Read measured evaluation; sample counts and tree counts are not AI IQ."""
        stats = self.trainer.get_training_stats()
        report = {}
        path = os.path.join(self.trainer.training_data_dir, "latest_evaluation.json")
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                report = json.load(handle)
        return {"stats": stats, "report": report,
                "model_exists": self.trainer.auto_integrate_model()}

    def _set_analysis_text(self, widget, text):
        widget.delete(1.0, tk.END)
        widget.insert(1.0, text)

    def update_intelligence_score(self, data):
        report = data.get('report', {})
        evaluation = report.get('evaluation', {})
        candidate = evaluation.get('candidate', {})
        self.intelligence_score_var.set("Evaluated" if candidate else "Not evaluated")
        self.intelligence_level_var.set("Candidate published" if report.get('published') else "No new model deployed")
        text = (f"Target: kill in next 2 seconds\n"
                f"Held-out demos: {len(report.get('validation_demos', []))}\n"
                f"Brier score (lower is better): {candidate.get('brier_score', 'unavailable')}\n"
                f"Average precision: {candidate.get('average_precision', 'unavailable')}")
        self._set_analysis_text(self.score_breakdown_text, text)

    def update_knowledge_depth(self, data):
        stats = data.get('stats', {})
        text = (f"Unique pre-event player snapshots: {stats.get('total_training_samples', 0):,}\n"
                f"Independent demo files: {stats.get('total_demos_processed', 0)}\n"
                f"Published model versions: {stats.get('model_versions', 0)}\n"
                f"Excluded legacy/invalid/duplicate samples: {stats.get('rejected_samples', {})}\n"
                "Kill-only history is excluded from future-kill training.")
        self._set_analysis_text(self.knowledge_depth_text, text)

    def update_prediction_confidence(self, data):
        evaluation = data.get('report', {}).get('evaluation', {})
        candidate = evaluation.get('candidate', {})
        baseline = evaluation.get('prevalence_baseline', {})
        text = (f"Candidate Brier / constant baseline: {candidate.get('brier_score', 'N/A')} / {baseline.get('brier_score', 'N/A')}\n"
                f"Top-player kill coverage: {candidate.get('top1_kill_coverage', 'N/A')}\n"
                f"Nearest-enemy coverage: {evaluation.get('nearest_enemy_top1_coverage', 'N/A')}\n"
                f"Action snapshots evaluated: {candidate.get('action_snapshots', 0)}\n"
                "Accuracy alone does not measure camera quality or calibrated confidence.")
        self._set_analysis_text(self.prediction_confidence_text, text)

    def update_tactical_expertise(self, data):
        self._set_analysis_text(self.tactical_expertise_text,
            "Map knowledge, damage prediction, clutch handling and trade kills have no separate evaluation.\n"
            "Their quality cannot be inferred from sample volume or the number of decision trees.\n"
            "Mesh visibility and angles are checked by the geometry layer.")

    def update_learning_progress(self, data):
        evaluation = data.get('report', {}).get('evaluation', {})
        improvement = evaluation.get('brier_improvement_over_incumbent')
        if improvement:
            text = (f"Mean per-demo Brier improvement over deployed model: {improvement['mean']:.5f}\n"
                    f"Grouped bootstrap 95% interval: [{improvement['ci95_low']:.5f}, {improvement['ci95_high']:.5f}]\n"
                    f"Held-out demos: {improvement['independent_demos']}\n")
        else:
            text = "No comparison with a compatible deployed model is available.\n"
        text += "Repeated selection can overfit a fixed validation set; reserve fresh demos for final testing."
        self._set_analysis_text(self.learning_progress_text, text)

    def update_knowledge_gaps(self, data):
        report = data.get('report', {})
        reasons = report.get('evaluation', {}).get('rejection_reasons', [])
        text = "\n".join(reasons) if reasons else "No rejected candidate recorded. This does not establish general tactical expertise."
        text += "\nCollect independent demos and evaluate held-out camera coverage, Brier score and average precision."
        self._set_analysis_text(self.knowledge_gaps_text, text)

    def analyze_map_distribution(self):
        from observer_training import prepare_prediction_samples
        records = self.trainer._load_all_training_data()
        rows, _ = prepare_prediction_samples(records)
        accepted = {(row['demo_id'], row['sample_tick'], row['player_id']) for row in rows}
        demo_stats = {}
        seen = set()
        for record in records:
            identity = (record.get('demo_id'), record.get('sample_tick'), record.get('player_id'))
            if identity not in accepted or identity in seen:
                continue
            seen.add(identity)
            map_name = record.get('gamestate', {}).get('map', {}).get('name', 'unknown')
            stats = demo_stats.setdefault(map_name, {'samples': 0, 'demo_files': set()})
            stats['samples'] += 1
            stats['demo_files'].add(record['demo_id'])
        return demo_stats, len(seen)

    def update_map_analysis(self, data):
        maps, total = self.analyze_map_distribution()
        lines = [f"Eligible snapshots: {total}"]
        for map_name, stats in sorted(maps.items()):
            lines.append(f"{map_name}: {stats['samples']} snapshots, {len(stats['demo_files'])} demos")
        lines.append("Per-map performance has not been separately measured.")
        self._set_analysis_text(self.map_analysis_text, "\n".join(lines))

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
        print("\n📚 LEGACY DATA (excluded from prediction training):")
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

            if training_results.get('success') and training_results.get('published'):
                print(f"✅ Training completed!")
                print(f"   - Accuracy: {training_results.get('accuracy', 0):.3f}")

                # Integrate
                if self.trainer.auto_integrate_model():
                    print("✅ Model integrated successfully")
                else:
                    print("⚠️  Model integration issues")
            elif training_results.get('success'):
                print(training_results.get('message', 'Candidate evaluated; no model deployed'))
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
                    print(f"Candidate Brier score: {training_results['evaluation']['candidate']['brier_score']:.4f}")
                    print("Model published" if training_results.get('published') else training_results.get('message'))
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
