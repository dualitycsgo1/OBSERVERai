import socket

HOST = "127.0.0.1"
PORT = 8085

def send_cmd(cmd: str):
    with socket.create_connection((HOST, PORT)) as s:
        # CS2 netcon kræver newline efter kommando
        s.sendall((cmd + "\n").encode("utf-8"))
        # prøv at læse lidt svar
        s.settimeout(1.0)
        try:
            data = s.recv(4096)
            print("REPLY:", data.decode("utf-8", errors="ignore"))
        except socket.timeout:
            print("No reply (but command likely accepted).")

if __name__ == "__main__":
    print("Sending 'echo hello_from_python'")
    send_cmd("echo hello_from_python")
