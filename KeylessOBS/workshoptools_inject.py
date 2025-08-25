"""
CS2 Auto Observer - Duel Detection (aggressiv v4.2)
Netcon-version: ingen tastetryk — sender 'spec_player "<navn>"' via TCP til -netconport.
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List, Tuple
import math
import os
import random
import socket
from collections import deque, defaultdict

# Virtual key codes for A,Q,E,W,U,I,O,P
HOTKEY_CODES = [0x41, 0x51, 0x45, 0x57, 0x55, 0x49, 0x4F, 0x50]

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("cs2_observer_advanced.log", encoding="utf-8"),
              logging.StreamHandler()]
)

# ---------------------------------------------------
# Netcon TCP-klient (CS2: -netconport <port>)
# ---------------------------------------------------
class NetconClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8085, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()

    def connect(self) -> bool:
        with self._lock:
            self.close()
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.settimeout(0.5)
                self._sock = s
                self._start_reader()
                logging.info(f"[netcon] Connected to {self.host}:{self.port}")
                return True
            except OSError as e:
                logging.warning(f"[netcon] Connect failed: {e}")
                self._sock = None
                return False

    def _start_reader(self):
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        while not self._stop_reader.is_set():
            try:
                with self._lock:
                    s = self._sock
                if not s:
                    time.sleep(0.2)
                    continue
                try:
                    data = s.recv(4096)
                    if not data:
                        # server closed
                        self._handle_disconnect("remote closed")
                        continue
                    # Hvis du vil, kan du printe svarene:
                    # logging.debug("[netcon] >> " + data.decode("utf-8", errors="ignore").rstrip())
                except socket.timeout:
                    continue
                except OSError as e:
                    self._handle_disconnect(f"recv error: {e}")
            except Exception as e:
                logging.debug(f"[netcon] reader loop: {e}")
                time.sleep(0.2)

    def _handle_disconnect(self, why: str):
        logging.warning(f"[netcon] Disconnected ({why})")
        self.close()

    def send_cmd(self, cmd: str) -> bool:
        line = (cmd + "\n").encode("utf-8")
        with self._lock:
            if not self._sock:
                # forsøg reconnect
                if not self.connect():
                    return False
            try:
                assert self._sock is not None
                self._sock.sendall(line)
                return True
            except OSError as e:
                logging.warning(f"[netcon] send failed ({e}), reconnecting...")
                self.close()
                if self.connect():
                    try:
                        assert self._sock is not None
                        self._sock.sendall(line)
                        return True
                    except OSError as e2:
                        logging.error(f"[netcon] send after reconnect failed: {e2}")
                        return False
                return False

    def close(self):
        self._stop_reader.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

# ---------------------------------------------------
# Hjælpefunktioner (geometri, parsing)
# ---------------------------------------------------
def parse_vec3(s: str) -> Optional[Tuple[float, float, float]]:
    try:
        # GSI kan være "x,y,z" eller "x, y, z"
        parts = [p.strip() for p in s.split(",")]
        x, y, z = map(float, parts[:3])
        return x, y, z
    except Exception:
        return None

def norm2d(v):
    l = math.hypot(v[0], v[1])
    if l == 0:
        return (0.0, 0.0)
    return (v[0] / l, v[1] / l)

def dot2d(a, b):
    return a[0] * b[0] + a[1] * b[1]

def distance2d(p1, p2) -> float:
    try:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    except Exception:
        return float("inf")

def clamp01(x):
    return max(0.0, min(1.0, x))

def angle_facing_score(player_pos, player_forward, target_pos):
    """
    0..1: 1.0 hvis man kigger direkte mod target (2D), 0 hvis 90+ grader væk.
    """
    try:
        px, py, _ = player_pos
        fx, fy, _ = player_forward
        tx, ty, _ = target_pos
        to_target = (tx - px, ty - py)
        f_n = norm2d((fx, fy))
        t_n = norm2d(to_target)
        dp = dot2d(f_n, t_n)  # [-1..1]
        return max(0.0, dp)
    except Exception:
        return 0.0

# ---------------------------------------------------
# Historik / Tracking
# ---------------------------------------------------
class History:
    def __init__(self, maxlen: int = 8):
        self.store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, steamid: str, pos, fwd, health: int, round_kills: int, t: float):
        self.store[steamid].append({"t": t, "pos": pos, "fwd": fwd, "hp": health, "rk": round_kills})

    def last2(self, sid: str):
        dq = self.store.get(sid)
        if not dq or len(dq) < 2:
            return None, None
        return dq[-2], dq[-1]

def relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr, dt_min=0.05) -> float:
    """Positiv værdi når spillerne nærmer sig hinanden."""
    if not p1_prev or not p1_curr or not p2_prev or not p2_curr:
        return 0.0
    dt1 = max(p1_curr["t"] - p1_prev["t"], dt_min)
    dt2 = max(p2_curr["t"] - p2_prev["t"], dt_min)
    pos1_prev = p1_prev["pos"]
    pos1_curr = p1_curr["pos"]
    pos2_prev = p2_prev["pos"]
    pos2_curr = p2_curr["pos"]
    if not pos1_prev or not pos1_curr or not pos2_prev or not pos2_curr:
        return 0.0
    v1 = ((pos1_curr[0] - pos1_prev[0]) / dt1, (pos1_curr[1] - pos1_prev[1]) / dt1)
    v2 = ((pos2_curr[0] - pos2_prev[0]) / dt2, (pos2_curr[1] - pos2_prev[1]) / dt2)
    rel_v = (v1[0] - v2[0], v1[1] - v2[1])
    r_now = (pos2_curr[0] - pos1_curr[0], pos2_curr[1] - pos1_curr[1])
    r_dir = norm2d(r_now)
    closing_speed = -dot2d(rel_v, r_dir)  # >0 = closing
    return closing_speed

def time_to_contact(distance: float, closing_speed: float, min_speed=1.0) -> float:
    spd = max(min_speed, closing_speed)
    return distance / spd

# ---------------------------------------------------
# Hovedklasse
# ---------------------------------------------------
class CS2DuelDetector:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.active = self.config.get("observer", {}).get("enable_auto_switch", True)

        # Netcon
        obs = self.config.get("observer", {})
        self.netcon_host = obs.get("netcon_host", "127.0.0.1")
        self.netcon_port = int(obs.get("netcon_port", 8085))
        self.netcon = NetconClient(self.netcon_host, self.netcon_port)

        # Cooldown/hold/rotation
        self.switch_cooldown = obs.get("switch_cooldown", 1.2)
        self.min_hold_time = self.config.get("hysteresis", {}).get("min_hold_time", 1.1)
        self.switch_margin = self.config.get("hysteresis", {}).get("switch_margin", 0.08)
        self.score_ema_alpha = self.config.get("hysteresis", {}).get("score_ema_alpha", 0.55)
        self.rotation_window = self.config.get("rotation", {}).get("window", 1.2)
        self.rotation_delta = self.config.get("rotation", {}).get("delta", 0.04)

        self.last_switch_time = datetime.min
        self.current_target_sid: Optional[str] = None
        self.current_target_start: Optional[datetime] = None
        self.current_target_score_ema: float = 0.0
        self._last_gamestate: Dict[str, Any] = {}

        # Duel detection
        dd = self.config.get("duel_detection", {})
        self.max_duel_distance = dd.get("max_distance", 1400.0)
        self.min_duel_distance = dd.get("min_distance", 75.0)
        self.facing_threshold = dd.get("facing_threshold", 0.35)
        self.height_diff_max = dd.get("height_difference_max", 220.0)
        self.ttc_max = dd.get("ttc_max", 3.2)

        # Freezetime tracking (vi skifter bare ikke i FT; ingen tastetryk mere)
        self.freezetime_start: Optional[datetime] = None
        self.numbers_lockout_seconds = 19  # bevarer lockout-logik (kan bruges som “ingen switch i 19s”)

        # Historik/trigger-trackere
        self.history = History(maxlen=8)
        self.last_rk: Dict[str, int] = {}
        self.kill_focus_until: float = 0.0
        self.kill_focus_sid: Optional[str] = None

        self.last_hp: Dict[str, int] = {}
        self.damage_focus_until: float = 0.0
        self.damage_focus_sid: Optional[str] = None

        # Anti-stall
        self.last_any_switch: float = 0.0
        self.failsafe_interval = 5.0

        # Prøv at connecte netcon med det samme
        self.netcon.connect()

    # ------------- Config --------------
    def load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logging.info("Konfiguration indlæst")
                return config
            else:
                logging.warning("Bruger default konfiguration (aggressiv)")
                return self.get_default_config()
        except Exception as e:
            logging.error(f"Fejl ved indlæsning af config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "observer": {
                "switch_cooldown": 1.2,
                "enable_auto_switch": True,
                "netcon_host": "127.0.0.1",
                "netcon_port": 8085
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
                "awp": 1.0, "ak47": 0.9, "m4a4": 0.88, "m4a1": 0.88,
                "rifle": 0.7, "smg": 0.48, "shotgun": 0.5,
                "pistol": 0.42, "knife": 0.2
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

    # ------------- Historik/Triggers --------------
    def update_history_and_triggers(self, allplayers: Dict[str, Any], now_ts: float):
        trig_cfg = self.config.get("triggers", {})
        kill_focus_time = trig_cfg.get("kill_focus_time", 2.2)
        damage_drop_hp = trig_cfg.get("damage_drop_hp", 35)
        damage_focus_time = trig_cfg.get("damage_focus_time", 1.7)

        for sid, p in allplayers.items():
            if sid == "total":
                continue
            state = p.get("state", {}) or {}
            hp = int(state.get("health", 0))
            rk = int(state.get("round_kills", 0))
            pos = parse_vec3(p.get("position", "")) if p.get("position") else None
            fwd = parse_vec3(p.get("forward", "")) if p.get("forward") else None

            self.history.update(sid, pos, fwd, hp, rk, now_ts)

            prev_rk = self.last_rk.get(sid, rk)
            if rk > prev_rk:
                self.kill_focus_sid = sid
                self.kill_focus_until = now_ts + kill_focus_time
                logging.info(f"[TRIGGER] Kill focus på {p.get('name','?')} i {kill_focus_time:.1f}s")
            self.last_rk[sid] = rk

            prev_hp = self.last_hp.get(sid, hp)
            if hp > 0 and prev_hp - hp >= damage_drop_hp:
                attacker_sid = self._guess_attacker(sid, allplayers)
                if attacker_sid:
                    self.damage_focus_sid = attacker_sid
                    self.damage_focus_until = now_ts + damage_focus_time
                    an = allplayers.get(attacker_sid, {}).get("name", "?")
                    vn = p.get("name", "?")
                    logging.info(f"[TRIGGER] Damage: {vn} tog {prev_hp-hp} dmg → fokus på {an} i {damage_focus_time:.1f}s")
            self.last_hp[sid] = hp

        if self.kill_focus_sid and now_ts > self.kill_focus_until:
            self.kill_focus_sid = None
        if self.damage_focus_sid and now_ts > self.damage_focus_until:
            self.damage_focus_sid = None

    def _guess_attacker(self, victim_sid: str, allplayers: Dict[str, Any]) -> Optional[str]:
        vic = allplayers.get(victim_sid, {})
        vic_team = vic.get("team", "")
        vic_pos = parse_vec3(vic.get("position", "")) if vic.get("position") else None
        if not vic_pos:
            return None
        best_sid = None
        best_score = 0.0
        for sid, p in allplayers.items():
            if sid in ("total", victim_sid):
                continue
            if p.get("team", "") == vic_team:
                continue
            pos = parse_vec3(p.get("position", "")) if p.get("position") else None
            fwd = parse_vec3(p.get("forward", "")) if p.get("forward") else None
            if not pos or not fwd:
                continue
            d = distance2d(pos, vic_pos)
            if d > 1200:
                continue
            fscore = angle_facing_score(pos, fwd, vic_pos)
            s = clamp01(0.6 * (1.0 - min(d / 1200.0, 1.0)) + 0.4 * fscore)
            if s > best_score:
                best_score = s
                best_sid = sid
        return best_sid

    # ------------- Scoring -------------
    def get_best_weapon_score(self, player_data: Dict[str, Any], weapon_scores: Dict[str, float]) -> Tuple[float, str, str]:
        weapons = player_data.get("weapons", {}) or {}
        best = 0.12
        best_name = "pistol"
        best_type = ""
        for w in weapons.values():
            name = (w.get("name", "") or "").lower()
            wtype = (w.get("type", "") or "").lower()
            if "knife" in name and len(weapons) > 1:
                continue
            sc = 0.12
            for key, val in weapon_scores.items():
                if key in name:
                    sc = val
                    break
            if "rifle" in wtype or "sniper" in wtype:
                sc += 0.1
            if sc > best:
                best = sc
                best_name = name
                best_type = wtype
        return best, best_name, best_type

    def weapon_distance_suitability(self, weapon_name: str, weapon_type: str, dist: float) -> float:
        wr = self.config.get("weapon_ranges", {})
        name = (
            "awp" if "awp" in weapon_name else
            "rifle" if ("rifle" in weapon_type or "m4" in weapon_name or "ak" in weapon_name) else
            "smg" if "smg" in weapon_type else
            "shotgun" if ("shotgun" in weapon_type or "nova" in weapon_name or "mag-7" in weapon_name) else
            "pistol"
        )
        r = wr.get(name, {"close": 200, "far": 700})
        if dist <= r["close"]:
            return clamp01(dist / max(1.0, r["close"]))
        elif dist >= r["far"]:
            return clamp01(r["far"] / max(1.0, dist))
        else:
            return 1.0

    def isolation_score(self, sid: str, allplayers: Dict[str, Any], radius_close=340.0) -> float:
        me = allplayers.get(sid, {})
        my_team = me.get("team", "")
        my_pos = parse_vec3(me.get("position", "")) if me.get("position") else None
        if not my_pos or not my_team:
            return 0.0
        allies = enemies = 0
        for osid, od in allplayers.items():
            if osid in ("total", sid):
                continue
            if od.get("state", {}).get("health", 0) <= 0:
                continue
            opos = parse_vec3(od.get("position", "")) if od.get("position") else None
            if not opos:
                continue
            if distance2d(my_pos, opos) <= radius_close:
                if od.get("team", "") == my_team:
                    allies += 1
                else:
                    enemies += 1
        if enemies >= 2 and allies <= 1:
            return 1.0
        if enemies == 1 and allies == 0:
            return 0.75
        if enemies >= 3:
            return 0.9
        if allies == 0 and enemies == 0:
            return 0.25
        return clamp01(0.35 + 0.16 * enemies - 0.1 * allies)

    def score_player_in_duel(self, sid_me: str, sid_op: str, allplayers: Dict[str, Any], dist: float, t: float) -> float:
        scw = self.config.get("scoring", {})
        wscores = self.config.get("weapon_scores", {})

        me = allplayers.get(sid_me, {})
        op = allplayers.get(sid_op, {})
        me_state = me.get("state", {}) or {}
        op_state = op.get("state", {}) or {}

        if me_state.get("health", 0) <= 0 or op_state.get("health", 0) <= 0:
            return 0.0

        me_pos = parse_vec3(me.get("position", "")) if me.get("position") else None
        op_pos = parse_vec3(op.get("position", "")) if op.get("position") else None
        me_fwd = parse_vec3(me.get("forward", "")) if me.get("forward") else None
        op_fwd = parse_vec3(op.get("forward", "")) if op.get("forward") else None

        o1 = angle_facing_score(me_pos, me_fwd, op_pos) if (me_pos and me_fwd and op_pos) else 0.0
        o2 = angle_facing_score(op_pos, op_fwd, me_pos) if (op_pos and op_fwd and me_pos) else 0.0
        orient_me = clamp01(0.65 * o1 + 0.35 * max(0.0, o1 - o2))

        me_hp = float(me_state.get("health", 0))
        op_hp = float(op_state.get("health", 0))
        health_score = clamp01(min(me_hp / max(op_hp, 1.0), 2.0) / 2.0)

        w_me, wname, wtype = self.get_best_weapon_score(me, wscores)
        weapon_quality = clamp01(w_me)
        dist_suit = self.weapon_distance_suitability(wname, wtype, dist)
        duel_prox = 1.0 if dist < 280 else (0.75 if dist < 600 else 0.45)

        rk = int(me_state.get("round_kills", 0))
        recent_perf = clamp01(min(0.22 * rk, 0.66))

        iso = self.isolation_score(sid_me, allplayers)

        p1_prev, p1_curr = self.history.last2(sid_me)
        p2_prev, p2_curr = self.history.last2(sid_op)
        closing = relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr)
        ttc = time_to_contact(dist, closing, min_speed=1.0)
        ttc_bonus = 1.0 if ttc <= self.ttc_max else max(0.0, (self.ttc_max / ttc))

        flashed = float(me_state.get("flashed", 0))
        blind_pen = 0.0
        if flashed >= 75:
            blind_pen = 0.45
        elif flashed >= 35:
            blind_pen = 0.2

        score = (
            scw.get("w_orientation_adv", 0.33) * orient_me
            + scw.get("w_distance_suit", 0.14) * dist_suit
            + scw.get("w_health", 0.12) * health_score
            + scw.get("w_weapon_quality", 0.14) * weapon_quality
            + scw.get("w_duel_proximity", 0.16) * duel_prox
            + scw.get("w_recent_perf", 0.06) * recent_perf
            + scw.get("w_isolation", 0.05) * iso
        )
        score *= clamp01(0.6 + 0.4 * ttc_bonus)
        score *= (1.0 - blind_pen)
        return clamp01(score)

    # ------------- Duel-finder -------------
    def find_potential_duels(self, allplayers: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        duels = []
        players = [(sid, data) for sid, data in allplayers.items() if sid != "total"]
        for i, (sid1, p1) in enumerate(players):
            for sid2, p2 in players[i + 1 :]:
                if p1.get("team", "") == p2.get("team", "") or not p1.get("team") or not p2.get("team"):
                    continue
                if p1.get("state", {}).get("health", 0) <= 0 or p2.get("state", {}).get("health", 0) <= 0:
                    continue
                pos1 = parse_vec3(p1.get("position", "")) if p1.get("position") else None
                pos2 = parse_vec3(p2.get("position", "")) if p2.get("position") else None
                if not pos1 or not pos2:
                    continue
                dist = distance2d(pos1, pos2)
                if dist > self.max_duel_distance or dist < self.min_duel_distance:
                    continue
                try:
                    if abs(pos1[2] - pos2[2]) > self.height_diff_max:
                        continue
                except Exception:
                    continue

                p1_prev, p1_curr = self.history.last2(sid1)
                p2_prev, p2_curr = self.history.last2(sid2)
                closing = relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr)
                if closing < 0.05 and dist > 600:
                    f1 = parse_vec3(p1.get("forward", "")) if p1.get("forward") else None
                    f2 = parse_vec3(p2.get("forward", "")) if p2.get("forward") else None
                    fscore = 0.0
                    if pos1 and f1 and pos2:
                        fscore = max(fscore, angle_facing_score(pos1, f1, pos2))
                    if pos2 and f2 and pos1:
                        fscore = max(fscore, angle_facing_score(pos2, f2, pos1))
                    if fscore < self.facing_threshold:
                        continue
                duels.append((sid1, sid2, dist))
        return duels

    # ------------- Valg og switching -------------
    def choose_best_player(self, gamestate: Dict[str, Any]) -> Tuple[Optional[str], float, Optional[str], float]:
        allplayers = gamestate.get("allplayers", {}) or {}
        duels = self.find_potential_duels(allplayers)
        if not duels:
            return None, 0.0, None, 0.0

        best_sid = None
        best_score = 0.0
        second_sid = None
        second_score = 0.0
        now_t = time.time()

        for sid1, sid2, dist in duels:
            s1 = self.score_player_in_duel(sid1, sid2, allplayers, dist, now_t)
            s2 = self.score_player_in_duel(sid2, sid1, allplayers, dist, now_t)

            for sid, sc in ((sid1, s1), (sid2, s2)):
                if sc > best_score:
                    second_sid, second_score = best_sid, best_score
                    best_sid, best_score = sid, sc
                elif sc > second_score and sid != best_sid:
                    second_sid, second_score = sid, sc

        return best_sid, best_score, second_sid, second_score

    def get_player_name(self, steamid: str, gamestate: Dict[str, Any]) -> Optional[str]:
        try:
            ap = gamestate.get("allplayers", {})
            p = ap.get(steamid)
            if not p:
                return None
            name = p.get("name") or ""
            name = self._sanitize_name(name)
            return name or None
        except Exception as e:
            logging.error(f"Fejl i navnslæsning: {e}")
            return None

            # --- FREEZETIME HÅNDTERING ---
            if is_freezetime:
                # Hvis vi lige gik ind i freezetime → tryk vilkårlig hotkey og start lockout
                if not self.freezetime_fkey_sent or self.last_freezetime_phase != 'freezetime':
                    try:
                        vk_hot = random.choice(HOTKEY_CODES)  # A,Q,E,W,U,I,O,P
                        user32.keybd_event(vk_hot, 0, 0, 0)
                        time.sleep(0.03)
                        user32.keybd_event(vk_hot, 0, 2, 0)
                        logging.info(f"Trykkede på hotkey '{chr(vk_hot)}' ved freezetime-start")
                    except Exception as e:
                        logging.error(f"Fejl ved hotkey tryk: {e}")
                    # markér FT-start og aktiver nummers-lockout
                    self.freezetime_fkey_sent = True
                    self.last_freezetime_phase = 'freezetime'
                    self.freezetime_start = datetime.now()
                # Under selve freezetime skifter vi ikke til spillere alligevel
                return
            else:
                # forlad freezetime: nulstil flag så vi kan trigge næste gang
                self.freezetime_fkey_sent = False
                self.last_freezetime_phase = None
                # self.freezetime_start beholdes – lockout varer 19s fra start uanset fase


    @staticmethod
    def _sanitize_name(name: str) -> str:
        # Fjern newlines, hardquotes og kontroltegn; vi sender altid med dobbelte-anførselstegn
        name = name.replace("\n", " ").replace("\r", " ")
        name = name.replace('"', "")  # fjern dobbelte anførselstegn
        # Klip til rimelig længde for sikkerhed
        return name.strip()[:64]

    def _numbers_lockout_active(self) -> bool:
        if not self.freezetime_start:
            return False
        return (datetime.now() - self.freezetime_start).total_seconds() < self.numbers_lockout_seconds

    def _can_switch_now(self) -> bool:
        # bevar samme hysterese/uafhængigt af tastetryk
        if self._numbers_lockout_active():
            return False
        now = datetime.now()
        if (now - self.last_switch_time).total_seconds() < self.switch_cooldown:
            return False
        if self.current_target_start and (now - self.current_target_start).total_seconds() < self.min_hold_time:
            return False
        return True

    def _should_switch_by_score(self, new_score: float) -> bool:
        cur = self.current_target_score_ema
        return new_score >= cur * (1.0 + self.switch_margin)

    def _update_current_target(self, sid: str, inst_score: float):
        a = self.score_ema_alpha
        if self.current_target_sid != sid:
            self.current_target_sid = sid
            self.current_target_start = datetime.now()
            self.current_target_score_ema = inst_score
        else:
            self.current_target_score_ema = a * inst_score + (1 - a) * self.current_target_score_ema

    def switch_to_player_name(self, name: str):
        """Send 'spec_player "<name>"' via netcon (ingen tastetryk)."""
        if not name:
            return
        if self._numbers_lockout_active():
            logging.debug("Numbers-lockout aktiv — ingen switch (FT-fenster).")
            return
        cmd = f'spec_player "{name}"'
        ok = self.netcon.send_cmd(cmd)
        if ok:
            self.last_switch_time = datetime.now()
            self.last_any_switch = time.time()
            logging.info(f"[switch] {cmd}")
        else:
            logging.error(f"[switch] FAILED sending: {cmd}")

    # ------------- Main processing -------------
    def process_gamestate(self, gamestate: Dict[str, Any]):
        try:
            if not self.active:
                return

            self._last_gamestate = gamestate
            allplayers = gamestate.get("allplayers", {}) or {}
            now_ts = time.time()
            self.update_history_and_triggers(allplayers, now_ts)

            # fase
            map_phase = (gamestate.get("map") or {}).get("phase")
            round_info = gamestate.get("round", {})
            phase_ends_in = round_info.get("phase_ends_in")
            is_freezetime = False
            try:
                if phase_ends_in is not None:
                    seconds_left = float(phase_ends_in)
                    if 19.5 <= seconds_left <= 20.5:
                        is_freezetime = True
            except Exception:
                pass

            if is_freezetime:
                # markér FT-start og aktiver "ingen switch i 19 sek" (bevarer din oprindelige lockout)
                if not self.freezetime_start:
                    self.freezetime_start = datetime.now()
                    logging.info("[phase] Freezetime start → lockout 19s for switch")
                return
            else:
                # bevar freezetime_start (lockout udløber selv)
                pass

            if self._numbers_lockout_active():
                return

            # Triggers har forrang
            trigger_sid = self.kill_focus_sid or self.damage_focus_sid
            if trigger_sid:
                if self._can_switch_now():
                    name = self.get_player_name(trigger_sid, gamestate)
                    if name:
                        self.switch_to_player_name(name)
                        self._update_current_target(trigger_sid, 0.9)
                return

            # Normal valg
            best_sid, best_score, second_sid, second_score = self.choose_best_player(gamestate)
            if not best_sid:
                if now_ts - self.last_any_switch > self.failsafe_interval and self._can_switch_now():
                    candidate = self._pick_any_live(allplayers)
                    if candidate:
                        name = self.get_player_name(candidate, gamestate)
                        if name:
                            self.switch_to_player_name(name)
                            self._update_current_target(candidate, 0.4)
                return

            if best_score < 0.28:
                if now_ts - self.last_any_switch > self.failsafe_interval and self._can_switch_now():
                    name = self.get_player_name(best_sid, gamestate)
                    if name:
                        self.switch_to_player_name(name)
                        self._update_current_target(best_sid, best_score)
                return

            # Rotation (når to spillere ligger tæt)
            if second_sid and (best_sid != second_sid) and abs(best_score - second_score) <= self.rotation_delta:
                if self._can_switch_now() and (datetime.now() - (self.current_target_start or datetime.min)).total_seconds() >= self.rotation_window:
                    pick = second_sid if self.current_target_sid == best_sid else best_sid
                    name = self.get_player_name(pick, gamestate)
                    if name:
                        self.switch_to_player_name(name)
                        self._update_current_target(pick, second_score if pick == second_sid else best_score)
                else:
                    if self.current_target_sid:
                        self._update_current_target(self.current_target_sid, max(0.0, self.current_target_score_ema * 0.95))
                return

            # Hysterese
            if (self.current_target_sid != best_sid and self._can_switch_now()
                and (self.current_target_sid is None or self._should_switch_by_score(best_score))):
                name = self.get_player_name(best_sid, gamestate)
                if name:
                    self.switch_to_player_name(name)
                    self._update_current_target(best_sid, best_score)
                return

            # Opdater EMA når vi allerede er på den bedste
            if self.current_target_sid == best_sid:
                self._update_current_target(best_sid, best_score)
            elif self.current_target_sid:
                self._update_current_target(self.current_target_sid, max(0.0, self.current_target_score_ema * 0.95))

        except Exception as e:
            logging.error(f"Fejl i gamestate processing: {e}")

    def _pick_any_live(self, allplayers: Dict[str, Any]) -> Optional[str]:
        for sid, p in allplayers.items():
            if sid == "total":
                continue
            if p.get("state", {}).get("health", 0) > 0:
                return sid
        return None

# ---------------------------------------------------
# HTTP server
# ---------------------------------------------------
class GameStateHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            gamestate = json.loads(post_data.decode("utf-8"))
            duel_detector.process_gamestate(gamestate)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except Exception as e:
            logging.error(f"Fejl i HTTP handler: {e}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_server(port: int = 8082):
    try:
        server_address = ("", port)
        httpd = HTTPServer(server_address, GameStateHandler)
        logging.info(f"Duel Detection Server startet på port {port}")
        httpd.serve_forever()
    except Exception as e:
        logging.error(f"Fejl ved start af server: {e}")

def control_interface():
    print("\n=== CS2 Auto Observer - Duel Detection (aggressiv v4.2) [netcon] ===")
    print("Kommandoer:")
    print("  'toggle'    - Tænd/sluk automatisk skift")
    print("  'status'    - Vis status")
    print("  'echo XYZ'  - Send 'echo XYZ' til CS2 konsol (test)")
    print("  'quit'      - Afslut program")
    print("  'cooldown X' / 'hold X' / 'margin X' justerer parametre\n")

    while True:
        try:
            cmd = input("> ").strip()
            low = cmd.lower()
            if low in ("quit", "exit"):
                break
            elif low == "toggle":
                duel_detector.active = not duel_detector.active
                print(f"Duel detection: {'AKTIVT' if duel_detector.active else 'INAKTIVT'}")
            elif low == "status":
                print(f"Status: {'AKTIVT' if duel_detector.active else 'INAKTIVT'}")
                print(f"Cooldown: {duel_detector.switch_cooldown}s")
                print(f"Min holdetid: {duel_detector.min_hold_time}s")
                print(f"Switch margin: {duel_detector.switch_margin}")
                print(f"Aktuel target: {duel_detector.current_target_sid} EMA={duel_detector.current_target_score_ema:.2f}")
                print(f"Kill focus: {duel_detector.kill_focus_sid} (t={max(0.0, duel_detector.kill_focus_until - time.time()):.1f}s)")
                print(f"Damage focus: {duel_detector.damage_focus_sid} (t={max(0.0, duel_detector.damage_focus_until - time.time()):.1f}s)")
            elif low.startswith("cooldown "):
                try:
                    duel_detector.switch_cooldown = float(low.split()[1])
                    print(f"Cooldown sat til {duel_detector.switch_cooldown}s")
                except Exception:
                    print("Ugyldig værdi")
            elif low.startswith("hold "):
                try:
                    duel_detector.min_hold_time = float(low.split()[1])
                    print(f"Min holdetid sat til {duel_detector.min_hold_time}s")
                except Exception:
                    print("Ugyldig værdi")
            elif low.startswith("margin "):
                try:
                    duel_detector.switch_margin = float(low.split()[1])
                    print(f"Switch-margin sat til {duel_detector.switch_margin}")
                except Exception:
                    print("Ugyldig værdi")
            elif low.startswith("echo "):
                msg = cmd[5:].strip()
                ok = duel_detector.netcon.send_cmd(f"echo {msg}")
                print("echo sendt" if ok else "echo fejlede (netcon ikke forbundet?)")
            else:
                print("Ukendt kommando")
        except KeyboardInterrupt:
            break

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    print("CS2 Auto Observer - Duel Detection (aggressiv v4.2) [netcon]")
    print("=============================================================")

    duel_detector = CS2DuelDetector()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    try:
        control_interface()
    except KeyboardInterrupt:
        logging.info("Program afbrudt")

    # ryd pænt op
    try:
        duel_detector.netcon.close()
    except Exception:
        pass

    print("Program afsluttet.")
