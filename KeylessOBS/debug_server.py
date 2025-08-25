"""
CS2 Auto Observer - Debug Version
Viser detaljeret information om HTTP requests og gamestate data
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import logging
import ctypes

# Setup logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs2_debug.log'),
        logging.StreamHandler()
    ]
)

class DebugGameStateHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Log incoming request
            client_ip = self.client_address[0]
            logging.info(f"HTTP POST modtaget fra {client_ip}")
            
            # Read data
            content_length = int(self.headers.get('Content-Length', 0))
            logging.info(f"Content-Length: {content_length} bytes")
            
            if content_length == 0:
                logging.warning("Tom HTTP request modtaget!")
                self.send_response(400)
                self.end_headers()
                return
            
            post_data = self.rfile.read(content_length)
            logging.info(f"Data modtaget: {len(post_data)} bytes")
            
            # Try to parse JSON
            try:
                gamestate = json.loads(post_data.decode('utf-8'))
                logging.info("✓ JSON parsing successful")
                
                # Log gamestate structure
                self.log_gamestate_info(gamestate)
                
                # Save raw data to file for inspection
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gamestate_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(gamestate, f, indent=2, ensure_ascii=False)
                logging.info(f"Gamestate data gemt til {filename}")
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing fejl: {e}")
                # Save raw data as text
                with open(f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'wb') as f:
                    f.write(post_data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK - Data received and logged')
            
        except Exception as e:
            logging.error(f"Fejl i HTTP handler: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_gamestate_info(self, gamestate):
        """Logger information om gamestate struktur"""
        logging.info("=== GAMESTATE DATA ANALYSE ===")
        
        # Top level keys
        top_keys = list(gamestate.keys())
        logging.info(f"Top level keys: {top_keys}")
        
        # Provider info
        provider = gamestate.get('provider', {})
        if provider:
            logging.info(f"Provider: {provider}")
        
        # Map info
        map_info = gamestate.get('map', {})
        if map_info:
            logging.info(f"Map: {map_info.get('name', 'Unknown')}")
        
        # Round info
        round_info = gamestate.get('round', {})
        if round_info:
            phase = round_info.get('phase', 'Unknown')
            logging.info(f"Round phase: {phase}")
        
        # Player info
        player = gamestate.get('player', {})
        if player:
            activity = player.get('activity', 'Unknown')
            logging.info(f"Player activity: {activity}")
        
        # All players info
        allplayers = gamestate.get('allplayers', {})
        if allplayers:
            player_count = len([k for k in allplayers.keys() if k != 'total'])
            logging.info(f"All players data: {player_count} spillere fundet")
            
            # Log first few players
            count = 0
            for steamid, player_data in allplayers.items():
                if steamid == 'total' or count >= 3:
                    continue
                name = player_data.get('name', 'Unknown')
                health = player_data.get('state', {}).get('health', 'N/A')
                logging.info(f"  Spiller: {name}, Health: {health}")
                count += 1
        else:
            logging.warning("INGEN allplayers data fundet!")
        
        # Phase countdowns
        phase_countdowns = gamestate.get('phase_countdowns', {})
        if phase_countdowns:
            logging.info(f"Phase countdowns: {phase_countdowns}")
        
        logging.info("=== END GAMESTATE ANALYSE ===")
    
    def log_message(self, format, *args):
        # Disable default HTTP logging
        pass

def start_debug_server(port: int = 8082):
    """Starter debug HTTP serveren"""
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, DebugGameStateHandler)
        logging.info(f"🔍 DEBUG SERVER startet på port {port}")
        logging.info("Alle HTTP requests og gamestate data vil blive logget!")
        logging.info("Vent på CS2 gamestate data...")
        httpd.serve_forever()
    except Exception as e:
        logging.error(f"Fejl ved start af debug server: {e}")

def debug_interface():
    """Debug kontrol interface"""
    print("\n=== CS2 DEBUG MODE ===")
    print("Dette program logger al HTTP trafik og gamestate data")
    print("Check 'cs2_debug.log' for detaljeret information")
    print("\nKommandoer:")
    print("  'quit' - Afslut program")
    print("  'status' - Vis server status")
    print("  'log' - Vis seneste log entries")
    print("\nServer lytter på port 8082...")
    print("Start CS2 og gå ind som observer!\n")
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if cmd == 'quit' or cmd == 'exit':
                logging.info("Debug session afsluttet")
                break
            elif cmd == 'status':
                print("Server kører og lytter på port 8082")
                print("Check cs2_debug.log for data")
            elif cmd == 'log':
                try:
                    with open('cs2_debug.log', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print("\nSeneste 10 log entries:")
                        for line in lines[-10:]:
                            print(line.strip())
                except:
                    print("Ingen log fil fundet endnu")
            else:
                print("Ukendt kommando. Prøv: quit, status, log")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Fejl i debug interface: {e}")

if __name__ == "__main__":
    print("CS2 Auto Observer - DEBUG MODE")
    print("==============================")
    
    # Start debug server i separat thread
    server_thread = threading.Thread(target=start_debug_server, daemon=True)
    server_thread.start()
    
    # Start debug interface
    try:
        debug_interface()
    except KeyboardInterrupt:
        logging.info("Debug program afbrudt af bruger")
    except Exception as e:
        logging.error(f"Uventet fejl: {e}")
    
    print("Debug session afsluttet.")
