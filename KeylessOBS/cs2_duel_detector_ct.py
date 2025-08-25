import json
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import ctypes

# Windows API setup for keyboard input
user32 = ctypes.windll.user32

# Virtual key codes for numbers 6-0
VK_CODES = {
    6: 0x36, 7: 0x37, 8: 0x38, 9: 0x39, 10: 0x30
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cs2_duel_detector_ct.log'),
        logging.StreamHandler()
    ]
)

class DuelDetectorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        gamestate = json.loads(post_data)

        # Process gamestate to determine key press
        player_slot = self.get_relevant_player_slot(gamestate)
        if player_slot in VK_CODES:
            self.press_key(VK_CODES[player_slot])

    def get_relevant_player_slot(self, gamestate):
        # Logic to determine relevant player slot (6-0)
        # Placeholder logic
        return 6

    def press_key(self, vk_code):
        user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk_code, 0, 2, 0)
        logging.info(f"Pressed key for player slot {vk_code - 0x30}")

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8084), DuelDetectorHandler)
    logging.info("Starting Duel Detector CT on port 8084")
    server.serve_forever()
