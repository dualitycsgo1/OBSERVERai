"""
Enkel Netværk Test Script
Tester om port 8082 er tilgængelig og kan modtage forbindelser
"""

import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK - Server is running!')
        print(f"✓ HTTP GET request modtaget fra {self.client_address[0]}")
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK - POST received!')
        
        content_length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(content_length)
        print(f"✓ HTTP POST request modtaget fra {self.client_address[0]}")
        print(f"  Data længde: {len(data)} bytes")
        if len(data) > 0:
            print(f"  Første 100 tegn: {data[:100]}")
    
    def log_message(self, format, *args):
        pass

def test_port_availability():
    """Tester om port 8082 er tilgængelig"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 8082))
        sock.close()
        print("✓ Port 8082 er tilgængelig")
        return True
    except OSError as e:
        print(f"✗ Port 8082 er ikke tilgængelig: {e}")
        print("  Måske kører et andet program allerede på denne port?")
        return False

def get_ip_address():
    """Finder lokal IP adresse"""
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        output = result.stdout
        
        lines = output.split('\n')
        for line in lines:
            if 'IPv4 Address' in line and '192.168' in line:
                ip_address = line.split(':')[-1].strip()
                return ip_address
        return None
    except:
        return None

def start_test_server():
    """Starter test HTTP server"""
    try:
        server_address = ('', 8082)
        httpd = HTTPServer(server_address, TestHandler)
        print("✓ Test server startet på port 8082")
        print("  Venter på HTTP requests fra CS2...")
        httpd.serve_forever()
    except Exception as e:
        print(f"✗ Fejl ved start af test server: {e}")

if __name__ == "__main__":
    print("CS2 Auto Observer - Enkel Netværk Test")
    print("======================================")
    print()
    
    # Test port availability
    if not test_port_availability():
        print("\nAndre programmer der måske bruger port 8082:")
        print("- Et andet CS2 observer program")
        print("- OBS Studio med gamestate integration")
        print("- Andre gamestate integration tools")
        print("\nLuk andre programmer og prøv igen.")
        input("Tryk Enter for at afslutte...")
        exit(1)
    
    # Show IP info
    ip_address = get_ip_address()
    if ip_address:
        print(f"✓ Din IP adresse: {ip_address}")
        print(f"  CS2 gamestate skal sende til: http://{ip_address}:8082")
    else:
        print("✗ Kunne ikke finde IP adresse automatisk")
    
    print("\n" + "="*50)
    print("TEST INSTRUKTIONER:")
    print("1. Lad denne server køre")
    print("2. Start CS2")
    print("3. Gå ind som observer/spectator i en aktiv match")
    print("4. Se om der kommer HTTP requests nedenfor")
    print("="*50)
    print()
    
    # Start server
    try:
        start_test_server()
    except KeyboardInterrupt:
        print("\nTest afsluttet af bruger.")
    except Exception as e:
        print(f"Uventet fejl: {e}")
    
    print("Test server stoppet.")
