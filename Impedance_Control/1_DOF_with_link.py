import os
import math
import time
import threading
import queue
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from dynamixel_sdk import *
import tkinter as tk
from tkinter import ttk
import numpy as np

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_VELOCITY = 128
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_CURRENT = 126
BAUDRATE = 1000000

PROTOCOL_VERSION = 2.0

DXL_ID = 2

DEVICENAME = '/dev/ttyUSB0'
# DEVICENAME = 'COM8'
TORQUE_ENABLE = 1  
TORQUE_DISABLE = 0  
DXL_MOVING_STATUS_THRESHOLD = 20

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel has been successfully connected")

def pos_to_rad(pos):
    return (pos / 4095) * 2 * np.pi

def vel_to_rad(vel):
    return (vel / 1023) * 2 * np.pi*0.229

def set_operating_mode(op):
    # time.sleep(0.1)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, 11, op)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    # time.sleep(0.1)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        print(f"Op:{op}")
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        print(f"Op:{op}")
    else:
        print("Dynamixel operating mode set successfully")

def main_loop(data_queue):
    global K_d, B_d, J_d, theta_d

    angles = []
    speeds = []
    torques = []
    theta_ds = []
    i = 0
    torque=0
    pos, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    while True:
        try:
            # set_operating_mode(3)
            position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
            if(position%4096>3215):
                packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                exit()
            if(position%4096<1000):
                packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                exit()
            if(abs((pos%4096)-(position%4096))>3000):
                position=pos
                time.sleep(0.1)
                packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, int(current))
                print("HELOO...............!!!!!!!!!!!!")
            else:
                pos=position%4096
            print(f"=======================================Position:{position}==============================\n")
            velocity, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_VELOCITY)
            print(f"=======================================Velocity:{velocity}==============================\n")
            load, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_CURRENT)
            print(f"=======================================Current:{load}==============================\n")

            theta = pos_to_rad(position % 4095)
            theta_dot = vel_to_rad(velocity % 1023)
            theta_dot_dot = (load * 2 * 2.69) / 1000
            print(f"Position:{theta}                      Error:{theta_d-theta}    Contrib:{K_d * (theta_d - theta)}")
            print(f"Velocity:{theta_dot}                  Error:{-theta_dot}       Contrib:{B_d * (0 - theta_dot)}")
            print(f"Acceleration:{theta_dot_dot}          Error:{-theta_dot_dot}   Contrib:{J_d * (0 - theta_dot_dot)}")
            if(torque<0):
                theta_dot=-1*theta_dot
            torque = J_d * 9.8*math.cos((((position%4096)-1024)/2048)*np.pi )+ B_d * (10 - theta_dot) + K_d * (theta_d - theta)
            current = (torque * 1000) / (2 * 2.69)
            print(int(current))
            set_operating_mode(0)
            packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, int(current))
            print(f"Current:{current}")
            angles.append(theta)
            speeds.append(theta_dot)
            torques.append(torque)
            theta_ds.append(theta_d)  

            # Put the data in the queue
            data_queue.put((angles, speeds, torques, theta_ds))

            time.sleep(0.01)
            i += 1
        except KeyboardInterrupt:
            break

    # Disable Dynamixel torque
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    # Close port
    portHandler.closePort()

def update_plot(data_queue, lines, theta_d_line):
    try:
        while not data_queue.empty():
            angles, speeds, torques, theta_ds = data_queue.get_nowait()

            lines[0].set_data(range(len(angles)), angles)
            lines[1].set_data(range(len(speeds)), speeds)
            lines[2].set_data(range(len(torques)), torques)
            theta_d_line.set_data(range(len(theta_ds)), theta_ds)

            for line in lines:
                line.axes.relim()
                line.axes.autoscale_view()

            theta_d_line.axes.relim()
            theta_d_line.axes.autoscale_view()

            plt.draw()
    except queue.Empty:
        pass
    root.after(100, update_plot, data_queue, lines, theta_d_line)

def update_parameters():
    global K_d, B_d, J_d, theta_d
    K_d = float(K_d_var.get())
    B_d = float(B_d_var.get())
    J_d = float(J_d_var.get())
    theta_d = pos_to_rad(float(theta_d_var.get()))

# Impedance parameters
K_d = 0.18
B_d = 0.0005
J_d = 0.004
theta_d = pos_to_rad(2048)

# Create the main window
root = tk.Tk()
root.title("Impedance Control Parameters")

K_d_var = tk.StringVar(value=str(K_d))
B_d_var = tk.StringVar(value=str(B_d))
J_d_var = tk.StringVar(value=str(J_d))
theta_d_var = tk.StringVar(value=str(2048))

tk.Label(root, text="K_d").grid(row=0, column=0)
tk.Entry(root, textvariable=K_d_var).grid(row=0, column=1)

tk.Label(root, text="B_d").grid(row=1, column=0)
tk.Entry(root, textvariable=B_d_var).grid(row=1, column=1)

tk.Label(root, text="J_d").grid(row=2, column=0)
tk.Entry(root, textvariable=J_d_var).grid(row=2, column=1)

tk.Label(root, text="theta_d").grid(row=3, column=0)
tk.Entry(root, textvariable=theta_d_var).grid(row=3, column=1)

tk.Button(root, text="Update", command=update_parameters).grid(row=4, columnspan=2)

# Initialize the plot
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
lines = []

for ax in axs:
    line, = ax.plot([], [])
    lines.append(line)

theta_d_line, = axs[0].plot([], [], 'r--', label='theta_d')
axs[0].legend()

axs[0].set_title("Angles")
axs[1].set_title("Speeds")
axs[2].set_title("Torques")

# Create a thread-safe queue
data_queue = queue.Queue()

# Start the main loop in a separate thread
main_thread = threading.Thread(target=main_loop, args=(data_queue,))
main_thread.daemon = True
main_thread.start()

# Update the plot
root.after(100, update_plot, data_queue, lines, theta_d_line)

# Run the GUI main loop
root.mainloop()
