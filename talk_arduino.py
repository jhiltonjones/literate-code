import serial.tools.list_ports
import time

def arduino_control(command, delay_us=None):
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()
    portsList = [one.device for one in ports]

    print(f"Available Ports: {portsList}")  
    com = "/dev/ttyACM0"  

    if com in portsList:
        serialInst.baudrate = 115200  # match Arduino
        serialInst.port = com
        serialInst.timeout = 1  # wait max 1 second for response
        serialInst.open()
        time.sleep(2)  # let Arduino settle after serial open
        serialInst.reset_input_buffer()
        print(f"Connected to {com}")
    else:
        print("Invalid port selected.")
        exit()

    full_command = command if delay_us is None else f"{command} {delay_us}"
    print(f"Sending command: {full_command}")
    serialInst.write((full_command + "\n").encode('utf-8'))  # send newline

    try:
        response = serialInst.readline().decode('utf-8').strip()
        if response:
            print(f"Arduino: {response}")
    except Exception as e:
        print(f"No response or error: {e}")

    serialInst.close()
    print("Connection closed.")

def distance_to_steps(distance_mm):
    mm_per_step = 0.118  # adjust if needed
    return int(distance_mm / mm_per_step)

if __name__ == '__main__':
    distance = 20  # mm
    steps = distance_to_steps(distance)
    delay_ms = 15  # 20 ms per half-step = 25 steps/sec
    arduino_control(f'ON {steps}', delay_us=delay_ms)
