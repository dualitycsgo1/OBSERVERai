"""
Netværk Test Script
Tester om port 8082 er tilgængelig og kan modtage forbindelser
"""

import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

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
        return False

def start_test_server():
    """Starter test HTTP server"""
    try:
        server_address = ('', 8082)
        httpd = HTTPServer(server_address, TestHandler)
        print("✓ Test server startet på port 8082")
        httpd.serve_forever()
    except Exception as e:
        print(f"✗ Fejl ved start af test server: {e}")

def test_local_connection():
    """Tester lokal forbindelse til serveren"""
    time.sleep(2)  # Vent på server start
    
    try:
        # Test GET request
        response = requests.get('http://localhost:8082', timeout=5)
        if response.status_code == 200:
            print("✓ Lokal GET test successful")
        else:
            print(f"✗ Lokal GET test fejlede: {response.status_code}")
    except Exception as e:
        print(f"✗ Lokal GET test fejlede: {e}")
    
    try:
        # Test POST request
        response = requests.post('http://localhost:8082', data='test data', timeout=5)
        if response.status_code == 200:
            print("✓ Lokal POST test successful")
        else:
            print(f"✗ Lokal POST test fejlede: {response.status_code}")
    except Exception as e:
        print(f"✗ Lokal POST test fejlede: {e}")

def test_external_connection():
    """Tester ekstern forbindelse"""
    import subprocess
    
    try:
        # Find IP adresse
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        output = result.stdout
        
        lines = output.split('\n')
        ip_address = None
        for line in lines:
            if 'IPv4 Address' in line and '192.168' in line:
                ip_address = line.split(':')[-1].strip()
                break
        
        if ip_address:
            print(f"Din IP adresse: {ip_address}")
            print(f"CS2 skal sende data til: http://{ip_address}:8082")
            
            try:
                response = requests.get(f'http://{ip_address}:8082', timeout=5)
                if response.status_code == 200:
                    print("✓ Ekstern forbindelse test successful")
                else:
                    print(f"✗ Ekstern forbindelse test fejlede: {response.status_code}")
            except Exception as e:
                print(f"✗ Ekstern forbindelse test fejlede: {e}")
        else:
            print("✗ Kunne ikke finde IP adresse")
            
    except Exception as e:
        print(f"✗ IP adresse test fejlede: {e}")

if __name__ == "__main__":
    print("CS2 Auto Observer - Netværk Test")
    print("=================================")
    print()
    
    # Test port availability
    if not test_port_availability():
        input("Tryk Enter for at afslutte...")
        exit(1)
    
    # Start server i separat thread
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    # Test connections
    test_thread = threading.Thread(target=test_local_connection, daemon=True)
    test_thread.start()
    
    # Wait a bit then test external
    time.sleep(3)
    test_external_thread = threading.Thread(target=test_external_connection, daemon=True)
    test_external_thread.start()
    
    print("\nServer kører nu på port 8082...")
    print("Test din CS2 gamestate integration ved at:")
    print("1. Starte CS2")
    print("2. Gå ind som observer i en match")
    print("3. Se om der kommer HTTP requests her")
    print("\nTryk Enter for at afslutte...")
    
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    print("Test afsluttet.")
