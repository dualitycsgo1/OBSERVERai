"""
CS2 Auto Observer - Automatisk spillerskift baseret på gamestate data
Lytter til gamestate integration data fra CS2 og skifter automatisk til spilleren
der mest sandsynligt får et kill.
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import keyboard
import logging
from typing import Dict, Any, Optional
import math
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs2_observer.log'),
        logging.StreamHandler()
    ]
)

class CS2GameState:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.last_gamestate = {}
        self.player_stats = {}
        self.current_round = 0
        self.last_switch_time = datetime.now()
        self.switch_cooldown = self.config.get('observer', {}).get('switch_cooldown', 2.0)
        self.active = self.config.get('observer', {}).get('enable_auto_switch', True)
        self.min_probability = self.config.get('observer', {}).get('min_kill_probability', 0.1)
        self.last_positions = {}
        self.current_watched_player = None
        self.player_switch_history = {}  # Track recent switches
        self.switch_decay_time = 10.0  # Seconds to reduce preference for recently watched players
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Indlæser konfiguration fra JSON fil"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logging.warning(f"Konfigurationsfil {config_path} ikke fundet. Bruger defaults.")
                return {}
        except Exception as e:
            logging.error(f"Fejl ved indlæsning af config: {e}")
            return {}
        
    def calculate_kill_probability(self, player_data: Dict[str, Any], gamestate: Dict[str, Any]) -> float:
        """
        Intelligent kill prediction baseret på:
        - Aktuelle engagements og skader
        - Position og proximity til fjender
        - Nylige aktiviteter og momentum
        - Våben og health status
        """
        try:
            score = 0.0
            player_name = player_data.get('name', 'Unknown')
            
            # Basic survival check
            state = player_data.get('state', {})
            health = state.get('health', 0)
            if health <= 0:
                return 0.0
            
            # 1. DAMAGE ACTIVITY SCORE (40% - mest vigtig!)
            damage_score = self.calculate_damage_activity_score(player_data, gamestate)
            score += damage_score * 0.40
            
            # 2. POSITION/ENGAGEMENT SCORE (25%)
            position_score = self.calculate_position_score(player_data, gamestate)
            score += position_score * 0.25
            
            # 3. EQUIPMENT SCORE (20%)
            equipment_score = self.calculate_equipment_score(player_data)
            score += equipment_score * 0.20
            
            # 4. MOMENTUM SCORE (15%)
            momentum_score = self.calculate_momentum_score(player_data)
            score += momentum_score * 0.15
            
            logging.debug(f"{player_name}: Dam={damage_score:.2f}, Pos={position_score:.2f}, Eq={equipment_score:.2f}, Mom={momentum_score:.2f}, Total={score:.2f}")
            
            return min(score, 1.0)
            
        except Exception as e:
            logging.error(f"Fejl i kill probability beregning: {e}")
            return 0.0
    
    def calculate_damage_activity_score(self, player_data: Dict[str, Any], gamestate: Dict[str, Any]) -> float:
        """Scorer baseret på nylige skader og engagements"""
        score = 0.0
        
        # Check for recent damage dealt (hvis tilgængeligt)
        state = player_data.get('state', {})
        
        # Flash/blind status - reducerer score betydeligt
        flashed = state.get('flashed', 0)
        if flashed > 200:  # Stærkt flashet
            score -= 0.5
        elif flashed > 50:  # Let flashet
            score -= 0.2
        
        # Burning (molotov damage) - reducerer score
        burning = state.get('burning', 0)
        if burning > 0:
            score -= 0.3
        
        # Scope status for sniper rifles
        weapons = player_data.get('weapons', {})
        for weapon_id, weapon in weapons.items():
            weapon_name = weapon.get('name', '').lower()
            if 'awp' in weapon_name or 'ssg08' in weapon_name:
                # Bonus hvis scoped med sniper
                # Dette er en approximation - der er ikke altid scope status i GSI
                score += 0.2
                break
        
        # Base activity score baseret på health ratio
        health = state.get('health', 100)
        if health < 50:  # Lavt HP kan indikere nyligt engagement
            score += 0.1
        
        return max(0.0, min(score + 0.3, 1.0))  # Base 0.3 + modifiers
    
    def calculate_position_score(self, player_data: Dict[str, Any], gamestate: Dict[str, Any]) -> float:
        """Scorer baseret på position og proximity til action"""
        score = 0.0
        
        # Position data (hvis tilgængeligt)
        position = player_data.get('position', {})
        if position:
            # Check om spilleren bevæger sig (indikerer aktiv play)
            # Vi kan ikke direkte måle hastighed, men vi kan se på koordinater
            x = position.get('x', 0)
            y = position.get('y', 0)
            z = position.get('z', 0)
            
            # Store disse til sammenligning næste gang (primitive movement detection)
            player_name = player_data.get('name', '')
            if hasattr(self, 'last_positions'):
                if player_name in self.last_positions:
                    old_x, old_y, old_z = self.last_positions[player_name]
                    distance_moved = ((x - old_x)**2 + (y - old_y)**2 + (z - old_z)**2)**0.5
                    if distance_moved > 50:  # Bevægelse indikerer aktivitet
                        score += 0.3
                    elif distance_moved < 10:  # Camping/holding angle
                        score += 0.1
                else:
                    self.last_positions = {}
                self.last_positions[player_name] = (x, y, z)
            else:
                self.last_positions = {player_name: (x, y, z)}
        
        # Team context
        team = player_data.get('team', '')
        if team:
            # Se på andre spilleres status på samme team
            allplayers = gamestate.get('allplayers', {})
            team_alive_count = 0
            for sid, p_data in allplayers.items():
                if sid != 'total' and p_data.get('team') == team:
                    if p_data.get('state', {}).get('health', 0) > 0:
                        team_alive_count += 1
            
            # Hvis få venner tilbage, højere chance for engagement
            if team_alive_count <= 2:
                score += 0.2
            elif team_alive_count <= 3:
                score += 0.1
        
        return min(score, 1.0)
    
    def calculate_equipment_score(self, player_data: Dict[str, Any]) -> float:
        """Scorer baseret på våben og equipment"""
        score = 0.0
        
        weapons = player_data.get('weapons', {})
        best_weapon_score = 0.0
        has_primary = False
        
        weapon_scores = {
            'awp': 1.0, 'ak47': 0.9, 'm4a4': 0.85, 'm4a1': 0.85,
            'famas': 0.6, 'galil': 0.6, 'aug': 0.7, 'sg553': 0.7,
            'deagle': 0.4, 'usp': 0.2, 'glock': 0.2, 'p250': 0.3,
            'tec9': 0.3, 'cz75': 0.3, 'mac10': 0.4, 'mp9': 0.4
        }
        
        for weapon_id, weapon in weapons.items():
            weapon_name = weapon.get('name', '').lower()
            weapon_type = weapon.get('type', '').lower()
            
            # Skip knives og granater
            if 'knife' in weapon_name or 'grenade' in weapon_name:
                continue
            
            # Find weapon score
            current_score = 0.1
            for w_name, w_score in weapon_scores.items():
                if w_name in weapon_name:
                    current_score = w_score
                    break
            
            if current_score > best_weapon_score:
                best_weapon_score = current_score
                if weapon_type in ['rifle', 'sniper']:
                    has_primary = True
            
            # Ammo check for primære våben
            if weapon_type in ['rifle', 'sniper']:
                ammo_clip = weapon.get('ammo_clip', 0)
                ammo_reserve = weapon.get('ammo_reserve', 0)
                if ammo_clip == 0 and ammo_reserve == 0:
                    current_score *= 0.1  # Drastisk reduktion hvis tom
                elif ammo_clip < 5:
                    current_score *= 0.7  # Reduktion hvis lav ammo
        
        score = best_weapon_score
        
        # Armor bonus
        state = player_data.get('state', {})
        armor = state.get('armor', 0)
        if armor > 50:
            score += 0.1
        elif armor > 0:
            score += 0.05
        
        # Helmet bonus
        helmet = state.get('helmet', False)
        if helmet:
            score += 0.05
        
        return min(score, 1.0)
    
    def calculate_momentum_score(self, player_data: Dict[str, Any]) -> float:
        """Scorer baseret på recent performance og momentum"""
        score = 0.0
        
        # Match stats
        match_stats = player_data.get('match_stats', {})
        kills = match_stats.get('kills', 0)
        deaths = match_stats.get('deaths', 1)  # Avoid division by zero
        assists = match_stats.get('assists', 0)
        
        # K/D ratio influence
        kd_ratio = kills / max(deaths, 1)
        if kd_ratio > 2.0:
            score += 0.4
        elif kd_ratio > 1.5:
            score += 0.3
        elif kd_ratio > 1.0:
            score += 0.2
        elif kd_ratio > 0.5:
            score += 0.1
        
        # Recent kills bonus
        if kills > 0:
            score += min(kills * 0.1, 0.3)
        
        # ADR (Average Damage per Round) approximation
        damage = match_stats.get('damage', 0)
        rounds_played = max(match_stats.get('score', 1), 1)
        adr = damage / rounds_played
        
        if adr > 100:
            score += 0.2
        elif adr > 75:
            score += 0.1
        
        # Money can indicate success
        money = player_data.get('state', {}).get('money', 0)
        if money > 8000:
            score += 0.1
        elif money > 5000:
            score += 0.05
        
        return min(score, 1.0)
    
    def get_best_player_to_watch(self, gamestate: Dict[str, Any]) -> Optional[str]:
        """Finder den spiller der mest sandsynligt får et kill med intelligent prediction"""
        try:
            allplayers = gamestate.get('allplayers', {})
            if not allplayers:
                return None
            
            best_player = None
            best_score = 0.0
            player_scores = []
            current_time = datetime.now()
            
            for steamid, player_data in allplayers.items():
                if steamid == 'total':  # Skip total stats
                    continue
                    
                base_score = self.calculate_kill_probability(player_data, gamestate)
                
                # Apply penalty for recently watched players
                adjusted_score = base_score
                if steamid in self.player_switch_history:
                    time_since_switch = (current_time - self.player_switch_history[steamid]).total_seconds()
                    if time_since_switch < self.switch_decay_time:
                        # Reduce score based on how recently we watched this player
                        penalty = (self.switch_decay_time - time_since_switch) / self.switch_decay_time
                        adjusted_score = base_score * (1.0 - penalty * 0.5)  # Up to 50% penalty
                
                player_name = player_data.get('name', 'Unknown')
                health = player_data.get('state', {}).get('health', 0)
                team = player_data.get('team', 'Unknown')
                
                player_scores.append((player_name, base_score, adjusted_score, health, team, steamid))
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_player = steamid
            
            # Sort og log top spillere for debugging
            player_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by adjusted score
            
            logging.info("=== KILL PREDICTION RANKING ===")
            for i, (name, base_score, adj_score, health, team, sid) in enumerate(player_scores[:5]):
                status = "💀" if health <= 0 else f"❤️{health}"
                penalty_indicator = f" (was {base_score:.3f})" if base_score != adj_score else ""
                logging.info(f"  {i+1}. {name} ({team}): {adj_score:.3f}{penalty_indicator} {status}")
            
            # Only switch if score is significantly better than current or above threshold
            if best_player and best_score > self.min_probability:
                # Extra check: if we recently switched to this player, require higher score
                if (self.current_watched_player == best_player and 
                    best_score < self.min_probability * 1.5):
                    logging.info(f"⏸️  Same player {allplayers[best_player].get('name', 'Unknown')}, needs higher score")
                    return None
                
                best_name = allplayers[best_player].get('name', 'Unknown')
                best_team = allplayers[best_player].get('team', 'Unknown')
                logging.info(f"🎯 SWITCHING TO: {best_name} ({best_team}) - Score: {best_score:.3f}")
                
                # Update switch history
                self.player_switch_history[best_player] = current_time
                self.current_watched_player = best_player
                
                return best_player
            else:
                logging.info(f"❌ No player above threshold ({self.min_probability:.2f})")
            
            return None
            
        except Exception as e:
            logging.error(f"Fejl i best player beregning: {e}")
            return None
    
    def get_player_slot(self, steamid: str, gamestate: Dict[str, Any]) -> Optional[int]:
        """Konverterer SteamID til slot nummer (1-10)"""
        try:
            allplayers = gamestate.get('allplayers', {})
            player_list = []
            
            for sid, player_data in allplayers.items():
                if sid == 'total':
                    continue
                player_list.append((sid, player_data.get('observer_slot', 0)))
            
            # Sorter efter observer_slot
            player_list.sort(key=lambda x: x[1])
            
            for i, (sid, slot) in enumerate(player_list):
                if sid == steamid:
                    return i + 1  # 1-based numbering
            
            return None
            
        except Exception as e:
            logging.error(f"Fejl i slot konvertering: {e}")
            return None
    
    def switch_to_player(self, slot: int):
        """Trykker på det rigtige talnummer for at skifte til spilleren"""
        try:
            now = datetime.now()
            if (now - self.last_switch_time).total_seconds() < self.switch_cooldown:
                return
            
            # Map slot til keyboard nummer
            key_map = {
                1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                6: '6', 7: '7', 8: '8', 9: '9', 10: '0'
            }
            
            if slot in key_map:
                key = key_map[slot]
                keyboard.press_and_release(key)
                logging.info(f"Skiftede til spiller slot {slot} (tast '{key}')")
                self.last_switch_time = now
            
        except Exception as e:
            logging.error(f"Fejl ved tastatur input: {e}")
    
    def process_gamestate(self, gamestate: Dict[str, Any]):
        """Behandler gamestate data og tager beslutning om spillerskift"""
        try:
            if not self.active:
                return
            
            # Check om det er et aktivt spil
            game_phase = gamestate.get('phase_countdowns', {})
            round_phase = gamestate.get('round', {}).get('phase')
            
            if round_phase not in ['live', 'freezetime']:
                logging.debug(f"Spil ikke aktivt (phase: {round_phase})")
                return
            
            # Find bedste spiller
            best_steamid = self.get_best_player_to_watch(gamestate)
            if not best_steamid:
                logging.debug("Ingen god spiller fundet")
                return
            
            # Konverter til slot
            slot = self.get_player_slot(best_steamid, gamestate)
            if not slot:
                logging.debug("Kunne ikke finde slot for spiller")
                return
            
            # Skift til spilleren
            self.switch_to_player(slot)
            
        except Exception as e:
            logging.error(f"Fejl i gamestate processing: {e}")

class GameStateHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            gamestate = json.loads(post_data.decode('utf-8'))
            
            # Process gamestate
            game_state.process_gamestate(gamestate)
            
            # Send response
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
            
        except Exception as e:
            logging.error(f"Fejl i HTTP handler: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Disable default HTTP logging
        pass

def start_server(port: int = 8082):
    """Starter HTTP serveren"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, GameStateHandler)
    logging.info(f"Server startet på port {port}")
    httpd.serve_forever()

def control_interface():
    """Simpel kontrol interface"""
    print("\n=== CS2 Auto Observer ===")
    print("Kommandoer:")
    print("  'toggle' - Tænd/sluk automatisk skift")
    print("  'status' - Vis status")
    print("  'quit' - Afslut program")
    print("  'cooldown X' - Sæt cooldown til X sekunder")
    print("\nTryk Enter for at starte...")
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if cmd == 'quit':
                logging.info("Afslutter program...")
                break
            elif cmd == 'toggle':
                game_state.active = not game_state.active
                status = "AKTIVT" if game_state.active else "INAKTIVT"
                print(f"Automatisk skift er nu: {status}")
            elif cmd == 'status':
                status = "AKTIVT" if game_state.active else "INAKTIVT"
                print(f"Status: {status}")
                print(f"Cooldown: {game_state.switch_cooldown} sekunder")
            elif cmd.startswith('cooldown '):
                try:
                    new_cooldown = float(cmd.split()[1])
                    game_state.switch_cooldown = new_cooldown
                    print(f"Cooldown sat til {new_cooldown} sekunder")
                except:
                    print("Ugyldig cooldown værdi")
            else:
                print("Ukendt kommando")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Fejl i control interface: {e}")

if __name__ == "__main__":
    # Global game state object
    game_state = CS2GameState()
    
    # Start HTTP server i separat thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Start control interface
    try:
        control_interface()
    except KeyboardInterrupt:
        logging.info("Program afbrudt af bruger")
    except Exception as e:
        logging.error(f"Uventet fejl: {e}")
11