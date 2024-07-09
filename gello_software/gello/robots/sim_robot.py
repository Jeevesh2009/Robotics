import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.robot import Robot

assert mujoco.viewer is mujoco.viewer

flag=0
def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def build_scene(robot_xml_path: str, gripper_xml_path: Optional[str] = None):
    # assert robot_xml_path.endswith(".xml")

    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)
    # arm_copy = mjcf.from_path(xml_path)

    if gripper_xml_path is not None:
        # attach gripper to the robot at "attachment_site"
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    return arena


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MujocoRobotServer:
    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        self._has_gripper = gripper_xml_path is not None
        arena = build_scene(xml_path, gripper_xml_path)

        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        # save xml_string to file
        with open("arena.xml", "w") as f:
            f.write(xml_string)

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu

        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def check_for_joint_limits(self,states):
        min_limits=np.array([-360,-118,-360,-20,-360,-97,-360])/180 
        max_limits=np.array([360,120,360,220,360,180,360])/180 
        # Convert states to a NumPy array
        states_arr = np.array(states[:7])
        
        # Check if every element in states is between min and max limits
        within_limits = np.all((states_arr >= min_limits) & (states_arr <= max_limits))
        if not within_limits:
            # Find indices where the limits are violated
            below_min_indices = np.where(states_arr < min_limits)[0]
            above_max_indices = np.where(states_arr > max_limits)[0]
            
            # Print the indices that are out of bounds
            if below_min_indices.size > 0:
                print(f"Indices below min limits: {below_min_indices}")
            if above_max_indices.size > 0:
                print(f"Indices above max limits: {above_max_indices}")
        return within_limits
      
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            # _joint_state[0:7]=0
            if(self.check_for_joint_limits(list(_joint_state))):
                self._joint_cmd = _joint_state
            else:
                # self.reset_to_initial_pos()
                # raise RuntimeError("Joint limit Reached")
                print("Joint Limit Reached")

      

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1
        flag=0
        x=0
        if(ee_pos[0]<0.15):
            flag=1
            x=1
        if(ee_pos[1]<-0.25):
            flag=1
        if(np.any(ee_pos>0.5)):
            flag=1
        elif(np.any(ee_pos<-0.5)):
            flag=1
        if(flag==1):
            # self.reset_to_initial_pos()
            if(x==1):
                print("Gripper is too close")
            # raise RuntimeError("Out of limits for workspace")
        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        print(ee_pos)
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

        # Cartesian impedance control gains.
        impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
        impedance_ori = np.asarray([150.0, 150.0, 150.0]) # [Nm/rad]

        # Joint impedance control gains.
        Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

        # Damping ratio for both Cartesian and joint impedance control.
        damping_ratio = 1.0

        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        Kpos: float = 0.95

        # Gain for the orientation component of the twist computation. This should be
        # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
        # orientation in one integration step.
        Kori: float = 0.95

        # Integration timestep in seconds.
        integration_dt: float = 1.0

        # Whether to enable gravity compensation.
        gravity_compensation: bool = False

        # Simulation timestep in seconds.
        dt: float = 0.002
        # Load the model and data.

        self._model.opt.timestep = dt

        # Compute damping and stiffness matrices.
        damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
        damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
        Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
        Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

        # # End-effector site we wish to control.
        site_name = "xarm7/link_tcp"
        site_id = self._model.site(site_name).id

        # Get the dof and actuator ids for the joints we wish to control. These are copied
        # from the XML file. Feel free to comment out some joints to see the effect on
        # the controller.
        joint_names = [
            "xarm7/joint1",
            "xarm7/joint2",
            "xarm7/joint3",
            "xarm7/joint4",
            "xarm7/joint5",
            "xarm7/joint6",
            "xarm7/joint7",
        ]
        dof_ids = np.array([self._model.joint(name).id for name in joint_names])
        actuator_ids = np.array([self._model.actuator(name).id for name in joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        key_name = "xarm7/home"
        key_id = self._model.key(key_name).id
        q0 = self._model.key(key_name).qpos[dof_ids]

        # # # Mocap body we will control with our mouse.
        # mocap_name = "xarm7/xarm_gripper_base_link"
        # mocap_id = self._model.body(mocap_name).mocapid[0]
        m = mujoco.MjModel.from_xml_path('third_party/mujoco_menagerie/ufactory_xarm7/xarm7.xml')
        d = mujoco.MjData(m)
        site2_name = "link_tcp"
        site2_id = m.site(site2_name).id
        # Pre-allocate numpy arrays.
        jac_temp = np.zeros((6, self._model.nv))
        jac=np.zeros((6,7))
        twist = np.zeros(6)
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        ee_quat=np.zeros(4)
        error_quat = np.zeros(4)
        M_inv_temp = np.zeros((self._model.nv, self._model.nv))
        M_inv=np.zeros((7,7))
        Mx = np.zeros((7, 7))
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                # Control algorithm
                desired_pos,desired_quat=self.compute_desired_pos(m,d,self._joint_cmd[:7],site2_id,ee_quat)
                # desired_pos=[0.24779234 , 0.00709713,  0.09694278]
                # desired_quat=[0,1,0,0]
                # print(desired_pos,desired_quat)
                dx = desired_pos - self._data.site(site_id).xpos
                twist[:3] = Kpos * dx / integration_dt
                mujoco.mju_mat2Quat(site_quat, self._data.site(site_id).xmat)
                print("Site_Quat",site_quat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, desired_quat, site_quat_conj)
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= Kori / integration_dt

                # Jacobian.
                mujoco.mj_jacSite(self._model, self._data, jac_temp[:3], jac_temp[3:], site_id)
                jac=jac_temp[:,[0,1,2,3,4,5,6]]
                # Compute the task-space inertia matrix.
                mujoco.mj_solveM(self._model, self._data, M_inv_temp, np.eye(self._model.nv))
                M_inv = M_inv_temp[np.ix_([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])]

                Mx_inv = jac @  M_inv @ jac.T
                if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                    Mx = np.linalg.inv(Mx_inv)
                else:
                    Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)



                # Compute generalized forces.
                tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ self._data.qvel[dof_ids]))

                # Add joint task in nullspace.

                # Add gravity compensation.
                if gravity_compensation:
                    tau += self._data.qfrc_bias[dof_ids]
                # Set the control signal and step the simulation.
                np.clip(tau, *self._model.actuator_ctrlrange[:7].T, out=tau)
                self._data.ctrl[actuator_ids] = tau[actuator_ids]
                self._data.ctrl[7]=self._joint_cmd[7]
                mujoco.mj_step(self._model, self._data)

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                # self._data.ctrl[:] = self._joint_cmd
                # self._data.qpos[:] = self._joint_cmd


                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos.copy()[: self._num_joints]

                # if self._print_joints:
                #     print(self._joint_state)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    # TODO remove?
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        self._data.time % 2
                    )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def compute_desired_pos(self, m,d,joint_angles,site_id,ee_quat):

        
        # with mujoco.viewer.launch_passive(m, d) as viewer:
        # # Close the viewer automatically after 30 wall-seconds.
        #     start = time.time()
        #     while viewer.is_running() and time.time() - start < 30:
        #         step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
        d.ctrl[:7]=joint_angles
        mujoco.mj_step(m, d)
        # time.sleep(0.02)
        ee_pos=d.site(site_id).xpos
        print("Pos",ee_pos)
        ee_mat=d.site(site_id).xmat
        print("Mat",ee_mat)
        print("Quat",ee_quat)
        mujoco.mju_mat2Quat(ee_quat, d.site(site_id).xmat)
        # Example modification of a viewer option: toggle contact points every two seconds.
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        # viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

        return ee_pos,ee_quat
    
    def reset_to_initial_pos(self):
        self.command_joint_state(np.zeros(self._num_joints))

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()
