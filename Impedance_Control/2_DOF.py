import os
import math
import time
import threading
import queue
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from dynamixel_sdk import *
# import dynamixel_sdk
import tkinter as tk
from tkinter import ttk
import numpy as np

# Handle platform-specific getch functionality
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

# Configuration Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_VELOCITY = 128
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_CURRENT = 126
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

DXL_IDS = [1, 2]  # IDs for two motors

DEVICENAME = '/dev/ttyUSB0'
# DEVICENAME = 'COM8'
TORQUE_ENABLE = 1  
TORQUE_DISABLE = 0  
DXL_MOVING_STATUS_THRESHOLD = 20

# Initialize PortHandler and PacketHandler
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
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print(f"Dynamixel {DXL_ID} has been successfully connected")

def pos_to_rad(pos):
    return (pos / 4095) * 2 * np.pi

def vel_to_rad(vel):
    return (vel / 1023) * 2 * np.pi * 0.229

def set_operating_mode(DXL_ID, op):
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, 11, op)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        print(f"Op:{op}")
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        print(f"Op:{op}")
    else:
        print(f"Dynamixel {DXL_ID} operating mode set successfully")

def main_loop(data_queue):
    global K_d, B_d, J_d,M_d, theta_d

    angles = [[], []]
    speeds = [[], []]
    torques = [[], []]
    theta_ds = [[], []]
    torque=[0,0]
    current = [0, 0]
    position=[0,0]
    velocity=[0,0]
    theta=[0,0]
    theta_dot=[0,0]
    theta_dot_dot=[0,0]
    acceleration=[0,0]
    prevvelocity=[0,0]
    prevposition=[0,0]
    while True:
        try:
            for i, DXL_ID in enumerate(DXL_IDS):
                position[i], _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
                velocity[i], _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_VELOCITY)
                acceleration[i]=(prevvelocity[i]-velocity[i]%1024)/0.01
                theta[i] = pos_to_rad(position[i] % 4095)
                theta_dot[i] = vel_to_rad(velocity[i] % 1024)
                theta_dot_dot[i]=vel_to_rad(acceleration[i]%1024)
                if prevposition[i]>position[i]:
                    theta_dot[i] = -theta_dot[i]
                torque[i] = J_d[i] * 9.8 * math.cos((((position[i] % 4096) - 1024) / 2048) * np.pi) + B_d[i] * (0 - theta_dot[i]) + K_d[i] * (theta_d[i] - theta[i])  + M_d[i]*(0-theta_dot_dot[i])
                current[i] = (torque[i] * 1000) / (2 * 2.69)
                print(f"Motor {DXL_ID} Current: {int(current[i])}")
                set_operating_mode(DXL_ID, 0)
                packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, int(current[i]))
                prevvelocity[i]=velocity[i]%1024
                prevposition[i]=position[i]
                angles[i].append(theta[i])
                speeds[i].append(theta_dot[i])
                torques[i].append(int(current[i]))
                theta_ds[i].append(theta_d[i])

            # Put the data in the queue
            data_queue.put((angles, speeds, torques, theta_ds))

            time.sleep(0.01)
        except KeyboardInterrupt:
            break

    # Disable Dynamixel torque and close port
    for DXL_ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()

def update_plot(data_queue, lines, theta_d_lines):
    try:
        while not data_queue.empty():
            angles, speeds, torques, theta_ds = data_queue.get_nowait()
            for i in range(2):
                lines[i][0].set_data(range(len(angles[i])), angles[i])
                lines[i][1].set_data(range(len(speeds[i])), speeds[i])
                lines[i][2].set_data(range(len(torques[i])), torques[i])
                theta_d_lines[i].set_data(range(len(theta_ds[i])), theta_ds[i])

                for line in lines[i]:
                    line.axes.relim()
                    line.axes.autoscale_view()

                theta_d_lines[i].axes.relim()
                theta_d_lines[i].axes.autoscale_view()

            plt.draw()
    except queue.Empty:
        pass
    root.after(100, update_plot, data_queue, lines, theta_d_lines)

def update_parameters():
    global K_d, B_d, J_d,M_d, theta_d
    K_d = [float(K_d_var[i].get()) for i in range(2)]
    B_d = [float(B_d_var[i].get()) for i in range(2)]
    J_d = [float(J_d_var[i].get()) for i in range(2)]
    M_d = [float(M_d_var[i].get()) for i in range(2)]
    theta_d = [pos_to_rad(float(theta_d_var[i].get())) for i in range(2)]

# Impedance parameters
K_d = [0.13, 0.11]
B_d = [0.02, 0.03]
M_d = [0.01 , 0.01]
J_d = [0.025, 0.0]
theta_d = [pos_to_rad(2048), pos_to_rad(2048)]

# Create the main window
root = tk.Tk()
root.title("Impedance Control Parameters")

K_d_var = [tk.StringVar(value=str(K_d[i])) for i in range(2)]
B_d_var = [tk.StringVar(value=str(B_d[i])) for i in range(2)]
J_d_var = [tk.StringVar(value=str(J_d[i])) for i in range(2)]
M_d_var = [tk.StringVar(value=str(M_d[i])) for i in range(2)]
theta_d_var = [tk.StringVar(value=str(2048)) for i in range(2)]

for i in range(2):
    tk.Label(root, text=f"K_d {i+1}").grid(row=0, column=i*2)
    tk.Entry(root, textvariable=K_d_var[i]).grid(row=0, column=i*2+1)

    tk.Label(root, text=f"B_d {i+1}").grid(row=1, column=i*2)
    tk.Entry(root, textvariable=B_d_var[i]).grid(row=1, column=i*2+1)

    tk.Label(root, text=f"M_d {i+1}").grid(row=2, column=i*2)
    tk.Entry(root, textvariable=M_d_var[i]).grid(row=2, column=i*2+1)

    tk.Label(root, text=f"J_d {i+1}").grid(row=3, column=i*2)
    tk.Entry(root, textvariable=J_d_var[i]).grid(row=3, column=i*2+1)

    tk.Label(root, text=f"theta_d {i+1}").grid(row=4, column=i*2)
    tk.Entry(root, textvariable=theta_d_var[i]).grid(row=4, column=i*2+1)

tk.Button(root, text="Update", command=update_parameters).grid(row=5, columnspan=5)

# Initialize the plot
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
lines = [[], []]

for i in range(2):
    for ax in axs:
        line, = ax.plot([], [])
        lines[i].append(line)

theta_d_lines = [axs[0].plot([], [], 'r--', label=f'theta_d {i+1}')[0] for i in range(2)]
axs[0].legend()

axs[0].set_title("Angles")
axs[1].set_title("Speed")
axs[2].set_title("Torque")

# Create a thread-safe queue
data_queue = queue.Queue()

# Start the main loop in a separate thread
main_thread = threading.Thread(target=main_loop, args=(data_queue,))
main_thread.daemon = True
main_thread.start()

# Update the plot
root.after(100, update_plot, data_queue, lines, theta_d_lines)

# Run the GUI main loop
root.mainloop()
