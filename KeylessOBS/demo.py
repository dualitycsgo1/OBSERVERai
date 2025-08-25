"""
CS2 Auto Observer - Demo/Test Version
Simulerer gamestate data for at teste funktionaliteten
"""

import json
import time
import random
from datetime import datetime
import ctypes
from ctypes import wintypes

# Windows API for keyboard
user32 = ctypes.windll.user32

# Virtual key codes
VK_CODES = {
    1: 0x31, 2: 0x32, 3: 0x33, 4: 0x34, 5: 0x35,
    6: 0x36, 7: 0x37, 8: 0x38, 9: 0x39, 10: 0x30
}

def send_key(vk_code):
    """Send a key press using Windows API"""
    try:
        user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk_code, 0, 2, 0)
        return True
    except:
        return False

def generate_fake_player_data():
    """Genererer falske spillerdata til test"""
    players = {}
    
    for i in range(10):
        steamid = f"player_{i+1}"
        players[steamid] = {
            "name": f"TestPlayer{i+1}",
            "observer_slot": i + 1,
            "state": {
                "health": random.randint(0, 100),
                "armor": random.randint(0, 100)
            },
            "weapons": {
                "0": {
                    "name": random.choice(["weapon_ak47", "weapon_m4a1", "weapon_awp", "weapon_glock"]),
                    "type": "Rifle",
                    "ammo_clip": random.randint(5, 30),
                    "ammo_clip_max": 30
                }
            },
            "match_stats": {
                "kills": random.randint(0, 3)
            }
        }
    
    return {
        "allplayers": players,
        "round": {"phase": "live"},
        "phase_countdowns": {"phase_ends_in": "45.2"}
    }

def calculate_kill_probability(player_data):
    """Beregner kill sandsynlighed"""
    health = player_data.get('state', {}).get('health', 0)
    if health <= 0:
        return 0.0
    
    score = (health / 100) * 0.3
    
    # Weapon scoring
    weapons = player_data.get('weapons', {})
    if weapons:
        weapon = list(weapons.values())[0]
        weapon_name = weapon.get('name', '').lower()
        if 'awp' in weapon_name:
            score += 0.4
        elif 'ak47' in weapon_name or 'm4' in weapon_name:
            score += 0.3
        else:
            score += 0.2
    
    # Kills bonus
    kills = player_data.get('match_stats', {}).get('kills', 0)
    score += min(kills * 0.1, 0.3)
    
    return min(score, 1.0)

def find_best_player(gamestate):
    """Finder bedste spiller"""
    allplayers = gamestate.get('allplayers', {})
    best_player = None
    best_score = 0.0
    
    print("\nSpiller analyse:")
    for steamid, player_data in allplayers.items():
        score = calculate_kill_probability(player_data)
        name = player_data.get('name', 'Unknown')
        health = player_data.get('state', {}).get('health', 0)
        kills = player_data.get('match_stats', {}).get('kills', 0)
        
        print(f"  {name}: Score {score:.2f} (HP: {health}, Kills: {kills})")
        
        if score > best_score and score > 0.15:
            best_score = score
            best_player = steamid
    
    if best_player:
        best_name = allplayers[best_player].get('name', 'Unknown')
        slot = allplayers[best_player].get('observer_slot', 1)
        print(f"\n>>> BEDSTE SPILLER: {best_name} (Slot {slot}, Score: {best_score:.2f})")
        return slot
    
    return None

def demo_mode():
    """Kører demo med simulerede data"""
    print("=== CS2 Auto Observer DEMO ===")
    print("Simulerer gamestate data...")
    print("Tryk Ctrl+C for at stoppe\n")
    
    round_num = 1
    
    try:
        while True:
            print(f"\n--- Runde {round_num} ---")
            
            # Generer falske data
            gamestate = generate_fake_player_data()
            
            # Find bedste spiller
            best_slot = find_best_player(gamestate)
            
            if best_slot and best_slot in VK_CODES:
                print(f"\nSender tastatur input: {best_slot}")
                vk_code = VK_CODES[best_slot]
                if send_key(vk_code):
                    key_name = str(best_slot) if best_slot != 10 else '0'
                    print(f"✓ Trykkede på tast '{key_name}'")
                else:
                    print("✗ Fejl ved tastatur input")
            else:
                print("\nIngen spiller fundet der opfylder kravene")
            
            print("\nVenter 5 sekunder til næste analyse...")
            time.sleep(5)
            round_num += 1
            
    except KeyboardInterrupt:
        print("\n\nDemo afsluttet!")

if __name__ == "__main__":
    print("Dette er en demo version til test af funktionalitet")
    print("Den simulerer CS2 gamestate data og tester tastatur input\n")
    
    choice = input("Vil du køre demo? (y/n): ").lower()
    if choice == 'y' or choice == 'yes':
        demo_mode()
    else:
        print("Demo afbrudt.")
