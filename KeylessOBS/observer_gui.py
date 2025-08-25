# observer_gui.py
# PyQt5 GUI til at starte/styre/stoppe cs2_duel_detector.py
# - Live kontrol: cooldown/hold/margin (sendes via stdin-kommandoer)
# - Øvrige settings via sliders/checkbox -> gemmes i config.json -> kan autogenstarte processen
# - Revert til defaults (matcher aggressive v4.2)
# - Viser live-log fra cs2_observer_advanced.log uden at blokere UI

import json
import os
import sys
import time
from pathlib import Path

from PyQt5.QtCore import (
    Qt, QTimer, QProcess
)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFormLayout, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox,
    QMessageBox, QPlainTextEdit
)

APP_TITLE = "CS2 Auto Observer – Control Panel"
SCRIPT_NAME = "cs2_duel_detector.py"
LOG_FILE = "cs2_observer_advanced.log"
CONFIG_FILE = "config.json"

# ----- Defaults (matcher aggressive v4.2) -----
DEFAULTS = {
    "observer": {
        "switch_cooldown": 1.2,
        "enable_auto_switch": True
    },
    "duel_detection": {
        "max_distance": 1400.0,
        "min_distance": 75.0,
        "facing_threshold": 0.35,
        "height_difference_max": 220.0,
        "ttc_max": 3.2
    },
    "scoring": {
        "w_orientation_adv": 0.33,
        "w_distance_suit": 0.14,
        "w_health": 0.12,
        "w_weapon_quality": 0.14,
        "w_duel_proximity": 0.16,
        "w_recent_perf": 0.06,
        "w_isolation": 0.05
    },
    "weapon_scores": {
        "awp": 1.0,
        "ak47": 0.9,
        "m4a4": 0.88,
        "m4a1": 0.88,
        "rifle": 0.7,
        "smg": 0.48,
        "shotgun": 0.5,
        "pistol": 0.42,
        "knife": 0.2
    },
    "weapon_ranges": {
        "awp":   {"close": 380, "far": 1300},
        "rifle": {"close": 240, "far": 950},
        "smg":   {"close": 160, "far": 520},
        "shotgun":{"close": 110, "far": 300},
        "pistol":{"close": 150, "far": 520}
    },
    "hysteresis": {
        "switch_margin": 0.08,
        "min_hold_time": 1.1,
        "score_ema_alpha": 0.55
    },
    "rotation": {
        "window": 1.2,
        "delta": 0.04
    },
    "triggers": {
        "kill_focus_time": 2.2,
        "damage_drop_hp": 35,
        "damage_focus_time": 1.7
    }
}

# --------- Utils ----------
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # deep copy
    return json.loads(json.dumps(DEFAULTS))

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def deep_get(cfg, path, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def deep_set(cfg, path, value):
    cur = cfg
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

def default_of(path):
    cur = DEFAULTS
    for k in path:
        cur = cur[k]
    return cur

# --------- Widget helper ----------
class LabeledSlider(QWidget):
    """
    Pæn slider med label + spinbox.
    Understøtter float (QDoubleSpinBox) og int (QSpinBox).
    """
    def __init__(self, title: str, path: list, cfg: dict,
                 minimum: float, maximum: float, step: float,
                 decimals: int = 2, integer: bool = False, live=False, parent=None):
        super().__init__(parent)
        self.path = path
        self.cfg = cfg
        self.integer = integer
        self.live = live  # om denne kan sendes live til processen

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight:600;")
        self.valueLabel = QLabel("")
        self.valueLabel.setMinimumWidth(70)

        # slider 0..N, vi mapper til [min,max]
        self._min = minimum
        self._max = maximum
        self._step = step
        self._decimals = decimals

        steps = max(1, int(round((maximum - minimum) / step)))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setFixedHeight(20)

        if integer:
            self.spin = QSpinBox()
            self.spin.setRange(int(minimum), int(maximum))
            self.spin.setSingleStep(max(1, int(round(step))))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(minimum, maximum)
            self.spin.setSingleStep(step)

        self.btnDefault = QPushButton("↺ Default")
        self.btnDefault.setFixedWidth(90)
        self.btnDefault.setToolTip("Sæt denne værdi tilbage til default")

        h1 = QHBoxLayout()
        h1.addWidget(self.title, 1)
        h1.addWidget(self.valueLabel)

        h2 = QHBoxLayout()
        h2.addWidget(self.slider, 1)
        h2.addWidget(self.spin)
        h2.addWidget(self.btnDefault)

        v = QVBoxLayout(self)
        v.addLayout(h1)
        v.addLayout(h2)

        # init værdi
        val = deep_get(cfg, path, default_of(path))
        self.set_value(val, update_cfg=False)

        # signaler
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        self.btnDefault.clicked.connect(self.set_default)

    def value(self):
        return deep_get(self.cfg, self.path, default_of(self.path))

    def set_value(self, val, update_cfg=True):
        # opdatér spin/slider
        if self.integer:
            val = int(round(val))
        else:
            val = round(float(val), self._decimals)

        # slider pos
        pos = int(round((val - self._min) / self._step))
        pos = max(self.slider.minimum(), min(self.slider.maximum(), pos))

        self.slider.blockSignals(True)
        self.spin.blockSignals(True)

        self.slider.setValue(pos)
        if self.integer:
            self.spin.setValue(int(val))
        else:
            self.spin.setValue(float(val))

        self.valueLabel.setText(str(val))

        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

        if update_cfg:
            deep_set(self.cfg, self.path, val)

    def set_default(self):
        self.set_value(default_of(self.path))

    def _slider_changed(self, pos):
        val = self._min + pos * self._step
        self.set_value(val)

    def _spin_changed(self, val):
        self.set_value(val)

# --------- Main window ----------
class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1920, 1080)

        # Initialize configuration
        self.cfg = load_config()

        # Apply Leetify-inspired color scheme
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1E1E2E"))  # Dark background
        palette.setColor(QPalette.WindowText, QColor("#FFFFFF"))  # White text
        palette.setColor(QPalette.Base, QColor("#2E2E3E"))  # Input fields
        palette.setColor(QPalette.Text, QColor("#FFFFFF"))
        palette.setColor(QPalette.Button, QColor("#3E3E5E"))  # Buttons
        palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
        self.setPalette(palette)

        # Internal state for log tailing
        self._log_pos = 0
        self._log_handle = None
        self.proc = None

        # Layout adjustments for 1920x1080
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # Top controls
        self.top_controls = QHBoxLayout()
        self.btnStart = QPushButton("Start Observer")
        self.btnStop = QPushButton("Stop Observer")
        self.statusLabel = QLabel("Status: Stoppet")
        self.statusLabel.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.top_controls.addWidget(self.btnStart)
        self.top_controls.addWidget(self.btnStop)
        self.top_controls.addWidget(self.statusLabel)
        self.main_layout.addLayout(self.top_controls)

        # Live control sliders
        self.live_group = self._build_live_group()
        self.main_layout.addWidget(self.live_group)

        # Config sliders
        self.config_group = self._build_config_group()
        self.main_layout.addWidget(self.config_group)

        # Log viewer
        self.logView = QPlainTextEdit()
        self.logView.setReadOnly(True)
        self.logView.setMinimumHeight(300)
        self.logView.setStyleSheet("background-color: #2E2E3E; color: #FFFFFF; font-family: Consolas;")
        self.main_layout.addWidget(self.logView)

        # Save and revert buttons
        self.save_revert_layout = QHBoxLayout()
        self.btnSave = QPushButton("Save Config")
        self.btnRevert = QPushButton("Revert to Defaults")
        self.save_revert_layout.addWidget(self.btnSave)
        self.save_revert_layout.addWidget(self.btnRevert)
        self.main_layout.addLayout(self.save_revert_layout)

        # signals
        self.btnStart.clicked.connect(self.start_observer)
        self.btnStop.clicked.connect(self.stop_observer)
        self.btnSave.clicked.connect(self.save_and_maybe_restart)
        self.btnRevert.clicked.connect(self.revert_all_defaults)

        # timers for log tail + heartbeat
        self.logTimer = QTimer(self)
        self.logTimer.timeout.connect(self.tail_log_once)
        self.logTimer.start(300)  # non-blocking, poll ny data

        self.procTimer = QTimer(self)
        self.procTimer.timeout.connect(self._poll_process)
        self.procTimer.start(700)

    # -------- Build groups ---------
    def _build_live_group(self) -> QGroupBox:
        """
        Parametre der kan ændres live via stdin-kommandoer i din v4.2:
        - cooldown X
        - hold X
        - margin X
        """
        box = QGroupBox("Live Control")
        box.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        form = QFormLayout()
        form.setSpacing(15)

        self.s_cooldown = LabeledSlider(
            "Switch cooldown (sek)", ["observer","switch_cooldown"], self.cfg,
            minimum=0.5, maximum=3.0, step=0.05, decimals=2, integer=False, live=True
        )
        self.s_hold = LabeledSlider(
            "Min. holdetid (sek)", ["hysteresis","min_hold_time"], self.cfg,
            minimum=0.5, maximum=3.0, step=0.05, decimals=2, integer=False, live=True
        )
        self.s_margin = LabeledSlider(
            "Switch‑margin (forbedring)", ["hysteresis","switch_margin"], self.cfg,
            minimum=0.00, maximum=0.30, step=0.01, decimals=2, integer=False, live=True
        )

        form.addRow(self.s_cooldown)
        form.addRow(self.s_hold)
        form.addRow(self.s_margin)

        # Når live‑sliders ændres → send kommando til processen, hvis den kører
        for w, cmd in (
            (self.s_cooldown, "cooldown"),
            (self.s_hold, "hold"),
            (self.s_margin, "margin"),
        ):
            w.slider.valueChanged.connect(lambda _=None, ww=w, cc=cmd: self._send_live_if_running(ww, cc))
            w.spin.valueChanged.connect(lambda _=None, ww=w, cc=cmd: self._send_live_if_running(ww, cc))

        box.setLayout(form)
        return box

    def _build_config_group(self) -> QGroupBox:
        """
        Øvrige parametre – kræver skriv til config.json; kan autogenstarte processen for at anvises
        """
        box = QGroupBox("Observer‑adfærd (kræver genstart for at anvises fuldt)")
        box.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        form = QFormLayout()
        form.setSpacing(15)

        # Korrekt kontrol for enable_auto_switch (var forkert i originalen)
        self.chk_enable_auto_switch = QCheckBox("Enable auto-switch")
        self.chk_enable_auto_switch.setChecked(bool(deep_get(self.cfg, ["observer","enable_auto_switch"], True)))
        self.chk_enable_auto_switch.stateChanged.connect(self._on_enable_auto_switch_changed)

        self.s_switch_cooldown_cfg = LabeledSlider(
            "Auto-switch cooldown (sek)", ["observer","switch_cooldown"], self.cfg,
            minimum=0.5, maximum=3.0, step=0.05, decimals=2, integer=False, live=False
        )
        self.s_max_distance = LabeledSlider(
            "Maks. afstand (duel)", ["duel_detection","max_distance"], self.cfg,
            minimum=100.0, maximum=3000.0, step=50.0, decimals=0, integer=True, live=False
        )
        self.s_min_distance = LabeledSlider(
            "Min. afstand (duel)", ["duel_detection","min_distance"], self.cfg,
            minimum=0.0, maximum=500.0, step=10.0, decimals=1, integer=False, live=False
        )
        self.s_facing_threshold = LabeledSlider(
            "Facing tærskel", ["duel_detection","facing_threshold"], self.cfg,
            minimum=0.0, maximum=1.0, step=0.01, decimals=2, integer=False, live=False
        )
        self.s_height_diff_max = LabeledSlider(
            "Højde forskel max", ["duel_detection","height_difference_max"], self.cfg,
            minimum=0.0, maximum=500.0, step=10.0, decimals=1, integer=False, live=False
        )
        self.s_ttc_max = LabeledSlider(
            "Max. tid-til-kollision", ["duel_detection","ttc_max"], self.cfg,
            minimum=0.1, maximum=10.0, step=0.1, decimals=1, integer=False, live=False
        )

        form.addRow(self.chk_enable_auto_switch)
        form.addRow(self.s_switch_cooldown_cfg)
        form.addRow(self.s_max_distance)
        form.addRow(self.s_min_distance)
        form.addRow(self.s_facing_threshold)
        form.addRow(self.s_height_diff_max)
        form.addRow(self.s_ttc_max)

        # Når config‑sliders ændres → gem til fil
        for w in (
            self.s_switch_cooldown_cfg,
            self.s_max_distance,
            self.s_min_distance,
            self.s_facing_threshold,
            self.s_height_diff_max,
            self.s_ttc_max,
        ):
            w.slider.valueChanged.connect(self._save_config_to_file)
            w.spin.valueChanged.connect(self._save_config_to_file)

        box.setLayout(form)
        return box

    # -------- Process control ---------
    def start_observer(self):
        if self.is_observer_running():
            self.statusLabel.setText("Status: Allerede kører")
            return

        # byg kommando
        cmd = [sys.executable, SCRIPT_NAME]
        cfg = self.cfg  # brug den aktuelle i memory

        # Tilføj CLI-arg for observer-sektionen
        # Hvis enable_auto_switch er False, medsend flag for at slå fra (uden værdi),
        # ellers medsend de parametre scriptet forventer.
        enable_auto = bool(deep_get(cfg, ["observer","enable_auto_switch"], True))
        if not enable_auto:
            cmd.append("--disable-auto-switch")
        # switch_cooldown sendes altid, så scriptet har en værdi
        sc = deep_get(cfg, ["observer","switch_cooldown"], default_of(["observer","switch_cooldown"]))
        cmd.extend(["--switch_cooldown", str(sc)])

        # Start som ATTACHED proces så vi kan skrive til stdin og læse stdout
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.readyReadStandardError.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_process_finished)

        self.proc.start(cmd[0], cmd[1:])
        started = self.proc.waitForStarted(3000)
        if not started:
            self._append_log_line("[GUI] Kunne ikke starte processen.")
            self.statusLabel.setText("Status: Fejl ved start")
            self.proc = None
            return

        self.statusLabel.setText("Status: Kører...")
        self._open_log_file(reset=True)
        self._append_log_line(f"[GUI] Startede {SCRIPT_NAME} med args: {' '.join(cmd[1:])}")

    def stop_observer(self):
        if not self.is_observer_running():
            self.statusLabel.setText("Status: Ikke kørende")
            return

        # forsøg pæn stop-kommando
        try:
            self.proc.write(b"stop\n")
            self.proc.flush()
        except Exception:
            pass

        # hvis den ikke selv stopper, så terminate/kill
        if not self.proc.waitForFinished(1500):
            self.proc.terminate()
            if not self.proc.waitForFinished(1500):
                self.proc.kill()
                self.proc.waitForFinished(1000)

        self.statusLabel.setText("Status: Stoppet")
        self.proc = None

    def is_observer_running(self) -> bool:
        return (self.proc is not None) and (self.proc.state() == QProcess.Running)

    def _poll_process(self):
        # hold statuslabel nogenlunde ajour
        if self.is_observer_running():
            self.statusLabel.setText("Status: Kører...")
        else:
            if self.proc is not None and self.proc.state() == QProcess.NotRunning:
                # allerede stoppet
                self.statusLabel.setText("Status: Stoppet")
                self.proc = None

    # -------- Log handling (non-blocking tail) ---------
    def _open_log_file(self, reset=False):
        try:
            if self._log_handle is None:
                self._log_handle = open(LOG_FILE, "r", encoding="utf-8", errors="replace")
                self._log_pos = 0
            if reset:
                self._log_handle.seek(0, os.SEEK_END)
                self._log_pos = self._log_handle.tell()
        except Exception as e:
            self._log_handle = None
            self._append_log_line(f"[GUI] Kunne ikke åbne logfil: {e}")

    def tail_log_once(self):
        # åbn fil hvis ikke åbnet
        if self._log_handle is None:
            if os.path.exists(LOG_FILE):
                self._open_log_file(reset=False)
            else:
                return

        try:
            self._log_handle.seek(self._log_pos)
            new_data = self._log_handle.read()
            if new_data:
                self._log_pos = self._log_handle.tell()
                # append linje for linje for at holde scroll pænt
                for line in new_data.splitlines():
                    self.logView.appendPlainText(line.rstrip("\n"))
                self.logView.verticalScrollBar().setValue(self.logView.verticalScrollBar().maximum())
        except Exception as e:
            self._append_log_line(f"[GUI] Log fejl: {e}")

    # -------- Process output ---------
    def _on_ready_read(self):
        if not self.is_observer_running():
            return
        try:
            # læs alle tilgængelige bytes
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if data:
                for line in data.splitlines():
                    self.logView.appendPlainText(line)
                self.logView.verticalScrollBar().setValue(self.logView.verticalScrollBar().maximum())
        except Exception as e:
            self._append_log_line(f"[GUI] Read fejl: {e}")

    def _on_process_finished(self, exitCode, exitStatus):
        self.statusLabel.setText(f"Status: Stoppet (Exit: {exitCode})")
        self.proc = None

    # -------- Live commands ---------
    def _send_live_if_running(self, widget: LabeledSlider, command: str):
        """Send 'command value' til processen via stdin, hvis den kører."""
        if not self.is_observer_running():
            return
        try:
            val = widget.value()
            msg = f"{command} {val}\n".encode("utf-8")
            self.proc.write(msg)
            self.proc.flush()
            self._append_log_line(f"[GUI] Sendt live: {command} {val}")
        except Exception as e:
            self._append_log_line(f"[GUI] Kunne ikke sende live-kommando: {e}")

    # -------- Config handling ---------
    def _on_enable_auto_switch_changed(self, state: int):
        deep_set(self.cfg, ["observer","enable_auto_switch"], state == Qt.Checked)
        self._save_config_to_file()

    def _save_config_to_file(self):
        """Gem nuværende cfg til config.json."""
        save_config(self.cfg)
        self._append_log_line("[GUI] Configuration saved to config.json")

    def save_and_maybe_restart(self):
        """Gem alt, og genstart processen hvis den kører."""
        # Alle LabeledSlider widgets skriver allerede værdier ind i self.cfg ved set_value()
        # Sørg lige for at få skrevet filen:
        self._save_config_to_file()

        # genstart hvis nødvendigt
        if self.is_observer_running():
            self._append_log_line("[GUI] Genstarter observer for at anvende nye settings...")
            self.stop_observer()
            # lille pause så port/file locks kan frigives
            QApplication.processEvents()
            time.sleep(0.5)
            self.start_observer()

    def revert_all_defaults(self):
        # Sæt alle værdier tilbage
        self.s_cooldown.set_default()
        self.s_hold.set_default()
        self.s_margin.set_default()

        self.chk_enable_auto_switch.setChecked(default_of(["observer","enable_auto_switch"]))
        self.s_switch_cooldown_cfg.set_default()
        self.s_max_distance.set_default()
        self.s_min_distance.set_default()
        self.s_facing_threshold.set_default()
        self.s_height_diff_max.set_default()
        self.s_ttc_max.set_default()

        self.save_and_maybe_restart()

    # -------- Helpers ---------
    def _append_log_line(self, text: str):
        self.logView.appendPlainText(text)
        self.logView.verticalScrollBar().setValue(self.logView.verticalScrollBar().maximum())

# -------- Entry Point --------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = ControlWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
