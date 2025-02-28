import serial.tools.list_ports

def arduino_control(command):
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()
    portsList = []

    for one in ports:
        portsList.append(one.device)
        print(one.device)  

    # com = input("Select the full port path for Arduino (e.g., /dev/ttyACM0): ")
    com = "/dev/ttyACM0"

    if com in portsList:
        serialInst.baudrate = 9600
        serialInst.port = com
        serialInst.open()
        serialInst.flush() 
        print(f"Connected to {com}")

    else:
        print("Invalid port selected.")
        exit()

    while True:
        # command = input("Arduino Command (ON/OFF/exit): ")
        
        serialInst.write((command + "\n").encode('utf-8'))  

        response = serialInst.readline().decode('utf-8').strip()
        print(f"Arduino: {response}") 
        serialInst.close()
        print("Connection closed") 
        break
        # if command.lower() == 'exit':
        #     serialInst.close()
        #     print("Connection closed.")
        #     break
def distance_arduino(distance):
    travel =  distance / 0.118
    return travel


if __name__ == '__main__':
    distance = 50
    travel = str(distance_arduino(distance))
    arduino_control(f'REV {travel}')


