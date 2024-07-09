import os
from dynamixel_sdk import *
import time
# Control table address for Dynamixel motors
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_TORQUE = 102
ADDR_PRESENT_TORQUE = 126
ADDR_OPERATING_MODE = 11

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL_ID = 2                # Dynamixel ID
BAUDRATE = 1000000         # Dynamixel default baudrate
DEVICENAME = '/dev/ttyUSB1'    # Check which port is being used on your controller

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()
else:
    print("open the port")

# Set port baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()
else:
    print("change the baudrate")


GOAL=10

# Enable Dynamixel Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque enabled for Dynamixel ID: %d" % DXL_ID)
time.sleep(100)
