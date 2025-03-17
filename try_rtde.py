import rtde_control

robot_ip = "192.168.56.1"
rtde_c = rtde_control.RTDEControlInterface(robot_ip)

# Test if the connection works
if rtde_c:
    print("Connected successfully!")
else:
    print("Connection failed.")
