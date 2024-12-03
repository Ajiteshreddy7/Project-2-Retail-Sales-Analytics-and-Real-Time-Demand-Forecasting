import socket
import time
import random
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_server(host='localhost', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        logging.info(f"Server started and listening on {host}:{port}")

        conn, address = server_socket.accept()
        logging.info(f"Connected to {address}")

        while True:
            product_id = random.randint(1, 100)
            quantity = random.choice([1, 2, 5, 10])  # Simulate more realistic purchasing patterns
            price = random.uniform(10.0, 200.0)  # Narrow the price range
            timestamp = int(time.time())
            transaction_record = f"{product_id},{quantity},{timestamp},{price:.2f}\n"
            conn.sendall(transaction_record.encode())
            time.sleep(random.uniform(0.5, 2))  # Vary the interval for a more dynamic simulation
    except KeyboardInterrupt:
        logging.info("Transaction data stream stopped by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        conn.close()
        server_socket.close()

if __name__ == '__main__':
    if len(sys.argv) > 2:
        start_server(sys.argv[1], int(sys.argv[2]))
    else:
        start_server()
