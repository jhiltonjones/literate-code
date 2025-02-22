import evdev
import time

device_path = "/dev/input/event262"  

try:
    controller = evdev.InputDevice(device_path)
    print(f"Using device: {controller.path} ({controller.name})")
except FileNotFoundError:
    print(f"Device not found at {device_path}. Check permissions and connection.")
    exit()

# Open a log file
log_file = "ps3_controller_log.txt"

print("Recording controller input. Press Ctrl+C to stop.")

try:
    with open(log_file, "w") as f:
        f.write("Timestamp, Type, Code, Value\n") 

        for event in controller.read_loop():
            timestamp = time.time()

            event_type = evdev.ecodes.EV[event.type] if event.type in evdev.ecodes.EV else f"Unknown({event.type})"

            if event.type in evdev.ecodes.bytype:
                event_code = evdev.ecodes.bytype[event.type].get(event.code, f"Unknown({event.code})")
            else:
                event_code = f"Unknown({event.code})"

            event_value = event.value
    
            log_entry = f"{timestamp}, {event_type}, {event_code}, {event_value}\n"
            f.write(log_entry)
            print(log_entry.strip())

except KeyboardInterrupt:
    print("\nRecording stopped.")
