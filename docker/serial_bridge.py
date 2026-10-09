"""Expose a host serial port over TCP, for Docker setups without USB passthrough (macOS + Colima).

    python serial_bridge.py /dev/tty.usbmodemXXXX [tcp_port]

Inside the control container, set SERIAL_TCP=host.docker.internal:<tcp_port> and
REAL_ROBOT_PORT=/dev/ttyBRIDGE0 in docker/.env.
"""
import socket
import sys
import threading

import serial

BAUDRATE = 1_000_000


def main():
    device = sys.argv[1]
    tcp_port = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
    ser = serial.Serial(device, BAUDRATE, timeout=0.001)

    server = socket.create_server(("0.0.0.0", tcp_port), reuse_port=True)
    print(f"Bridging {device} @ {BAUDRATE} <-> tcp :{tcp_port}")
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Client connected: {addr}")
        ser.reset_input_buffer()

        def serial_to_tcp():
            try:
                while True:
                    data = ser.read(ser.in_waiting or 1)
                    if data:
                        conn.sendall(data)
            except OSError:
                pass

        threading.Thread(target=serial_to_tcp, daemon=True).start()
        try:
            while data := conn.recv(4096):
                ser.write(data)
        except OSError:
            pass
        conn.close()
        print("Client disconnected")


if __name__ == "__main__":
    main()
