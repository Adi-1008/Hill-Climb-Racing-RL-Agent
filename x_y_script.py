import pydirectinput
import time

print("Tracking mouse coordinates. Press Ctrl+C in the terminal to stop.")
try:
    while True:
        print("script started")
        time.sleep(10)
        x, y = pydirectinput.position()
        print(f"X: {x}, Y: {y}")
except KeyboardInterrupt:
    print("Done tracking.")