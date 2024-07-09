import time
import numpy as np
from scipy import signal
from dynamixel_sdk import *

# Ensure the `control` library is installed
try:
    import control as ctrl
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "control"])
    import control as ctrl

# Control table address
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_VELOCITY = 128
ADDR_GOAL_TORQUE = 102

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL_ID = 1
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

# Enable Dynamixel torque
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1)

# Desired impedance parameters
J_d = 1.0  # Desired inertia
B_d = 0.1  # Desired damping
K_d = 10.0  # Desired stiffness

# Discrete-time controller parameters
sampling_time = 0.01  # Sampling time for the discrete controller

# Transfer function of the desired impedance in continuous time
num = [J_d, B_d, K_d]
den = [1, 0, 0]
system = ctrl.TransferFunction(num, den)

# Discretize the system
discrete_system = ctrl.sample_system(system, sampling_time)

# Extract discrete system coefficients
b, a = signal.tf2zpk(discrete_system.num[0][0], discrete_system.den[0][0])

# Desired position
theta_des = 2048  # Center position for Dynamixel

# Controller state
prev_error = 0
integral = 0

while True:
    # Read current position and velocity
    position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    velocity, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_VELOCITY)

    # Convert to radians and rad/s if needed
    theta = position
    theta_dot = velocity

    # Calculate error
    error = theta_des - theta

    # Implement PID control in discrete time
    integral += error * sampling_time
    derivative = (error - prev_error) / sampling_time
    prev_error = error

    # PID output
    torque = b[0]*error + b[1]*integral + b[2]*derivative - a[1]*prev_error - a[2]*integral

    # Send torque command to Dynamixel motor
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_TORQUE, int(torque))

    # Small delay for the control loop
    time.sleep(sampling_time)

# Disable Dynamixel torque
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)

# Close port
portHandler.closePort()
