import cv2
import socket
import numpy as np

def run_receiver(port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('0.0.0.0', port))
    print(f"Listening for BFMC Car Debug on port {port}...")

    try:
        while True:
            data, addr = server_socket.recvfrom(65536)
            
            if data.startswith(b'IMG'):
                img_data = data[3:] # Remove header
                
                # Decode JPEG
                np_arr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    cv2.imshow("BFMC Live Debugger", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_receiver()