import cv2
import socket
import struct
import pickle
import subprocess
import time
import argparse


def get_wsl_ip():
    try:
        ip = subprocess.check_output("wsl hostname -I", shell=True).decode().strip()
        return ip
    except Exception:
        return None

def connect_to_wsl(wsl_ip, port):
    while True:
        try:
            print(f"[INFO] Connecting to {wsl_ip}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((wsl_ip, port))
            print("[INFO] Connected!")
            return sock
        except Exception as e:
            print(f"[WARN] Connection failed: {e}. Retrying in 3s...")
            time.sleep(3)

def send_frames_forever(video_id, port):
    cap = cv2.VideoCapture(video_id)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    wsl_ip = get_wsl_ip()
    if not wsl_ip:
        print("Could not get WSL IP.")
        return

    client_socket = connect_to_wsl(wsl_ip, port)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            data = pickle.dumps(frame)
            size = struct.pack("!I", len(data))
            client_socket.sendall(size + data)

        except (BrokenPipeError, ConnectionResetError, socket.error) as e:
            print(f"[ERROR] Connection lost: {e}. Reconnecting...")
            client_socket.close()
            client_socket = connect_to_wsl(wsl_ip, port)
        except KeyboardInterrupt:
            print("[INFO] Exiting sender...")
            break

    cap.release()
    client_socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", default=0, type=int, help="Camera ID")
    parser.add_argument("--port", default=8485, type=int, help="Port number")
    args = parser.parse_args()
    send_frames_forever(args.video_id, args.port)

if __name__ == "__main__":
    main()