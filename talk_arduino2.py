# talk_arduino.py

import serial
import time
from queue import Queue
import threading

class ArduinoSerial:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=1):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2)
        self.serial.reset_input_buffer()
        print(f"[Arduino] Connected on {port}")

    def send_command(self, command, delay_us=None):
        full_command = command if delay_us is None else f"{command} {delay_us}"
        self.serial.write((full_command + "\n").encode("utf-8"))
        try:
            response = self.serial.readline().decode("utf-8").strip()
            if response:
                print(f"[Arduino] Response: {response}")
        except Exception as e:
            print(f"[Arduino] Error: {e}")

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("[Arduino] Serial closed.")

# Singleton pattern
arduino = ArduinoSerial()
arduino_queue = Queue()

def arduino_worker():
    while True:
        item = arduino_queue.get()
        if item is None:
            break
        command, delay = item
        arduino.send_command(command, delay)
        arduino_queue.task_done()

def distance_to_steps(distance_mm):
    mm_per_step = 0.118
    return int(distance_mm / mm_per_step)
def shutdown_arduino():
    arduino_queue.put(None)
    arduino.close()
arduino_thread = threading.Thread(target=arduino_worker, daemon=True)
arduino_thread.start()

if __name__ == '__main__':
    distance = 2# mm
    steps = distance_to_steps(distance)
    arduino_queue.put((f'REV {steps}', 20))  # queue the command

    arduino_queue.join()
    shutdown_arduino()    
