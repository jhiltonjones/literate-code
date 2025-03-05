import serial.tools.list_ports
import time

def arduino_control(command):
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()
    portsList = [one.device for one in ports]

    print(f"Available Ports: {portsList}")  
    com = "/dev/ttyACM1"  

    if com in portsList:
        serialInst.baudrate = 9600
        serialInst.port = com
        serialInst.open()
        time.sleep(3) 
        serialInst.flush()
        print(f"Connected to {com}")
    else:
        print("Invalid port selected.")
        exit()

    print(f"Sending command: {command}")
    serialInst.write((command + "\r\n").encode('utf-8'))  

    time.sleep(1)  
    response = serialInst.readline().decode('utf-8').strip()
    print(f"Arduino: {response}")

    serialInst.close()
    print("Connection closed.")

def distance_arduino(distance):
    travel = distance / 0.118
    return travel

if __name__ == '__main__':
    distance = 10
    travel = str(distance_arduino(distance))
    arduino_control(f'ON {travel}')  
