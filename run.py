"""
Run the Resume-Based Job Prediction app on localhost.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --host 0.0.0.0 --port 5000
"""
import argparse
import socket
from app import app


def find_available_port(start_port: int = 5002, end_port: int = 5100) -> int:
    """Find a free port between start_port and end_port."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start_port} and {end_port}")


def main():
    parser = argparse.ArgumentParser(description="Run the Resume-Based Job Prediction server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5002, help="Port to listen on (default: 5002)")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode (default: True)")
    args = parser.parse_args()

    port = find_available_port(args.port, args.port + 50)
    print(f"Starting server at http://{args.host}:{port}")
    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
