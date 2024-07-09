import mujoco
import mujoco.viewer
import time
import numpy as np


def numerical_diff(matrix, dt=1.0):
    """
    Perform numerical differentiation on a matrix using finite differences.
    
    Parameters:
    matrix (np.ndarray): Input matrix to differentiate.
    dt (float): Time step or spacing between samples.
    
    Returns:
    np.ndarray: Differentiated matrix.
    """
    # Central difference for interior points, forward/backward difference for boundaries
    diff_matrix = np.zeros_like(matrix)
    
    # Differentiation along rows (axis 0)
    diff_matrix[1:-1, :] = (matrix[2:, :] - matrix[:-2, :]) / (2 * dt)
    diff_matrix[0, :] = (matrix[1, :] - matrix[0, :]) / dt
    diff_matrix[-1, :] = (matrix[-1, :] - matrix[-2, :]) / dt
    
    # Differentiation along columns (axis 1)
    diff_matrix[:, 1:-1] = (matrix[:, 2:] - matrix[:, :-2]) / (2 * dt)
    diff_matrix[:, 0] = (matrix[:, 1] - matrix[:, 0]) / dt
    diff_matrix[:, -1] = (matrix[:, -1] - matrix[:, -2]) / dt
    
    return diff_matrix




def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    dt: float = 0.002
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("xarm7_nohand.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id
    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Pre-allocate numpy arrays.
    jac_temp = np.zeros((6, model.nv))
    M_inv_temp=np.zeros((model.nv,model.nv))
    # diag = damping * np.eye(6)
    # eye = np.eye(model.nv)
    # twist = np.zeros(6)
    # site_quat = np.zeros(4)
    # site_quat_conj = np.zeros(4)
    # error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE


        #Desired Position
        x_d=np.array([0.4,0,0.2])


        #Desired Velocity
        v_d=np.zeros(3)

        M_d=0.0015*np.eye(3)
        K_d=0.002*np.eye(3)
        B_d=0.001*np.eye(3)
        while viewer.is_running():
            step_start = time.time()

            
            #joint positions and velocities
            theta = data.qpos[dof_ids]
            theta_dot=data.qvel[dof_ids]

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac_temp[:3], jac_temp[3:], site_id)
            jac=jac_temp[:,[0,1,2,3,4,5,6]]
            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv_temp ,np.eye(model.nv))
            M_inv = M_inv_temp[np.ix_([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])]
            # Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(M_inv)) >= 1e-2:
                Mx = np.linalg.inv(M_inv)
            else:
                Mx = np.linalg.pinv(M_inv, rcond=1e-2)


            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            # Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(M_inv)) >= 1e-2:
                Mx = np.linalg.inv(M_inv)
            else:
                Mx = np.linalg.pinv(M_inv, rcond=1e-2)


            # CS positions and velocities
            position=data.site(site_id).xpos.copy()
            velocity=jac[:3] @ theta_dot
            


            #Errors in pos and velocity
            pos_error = x_d - position
            vel_error = v_d - velocity

            jac_inv=np.linalg.pinv(jac)             #J psuedo inv
            jac_dot=numerical_diff(jac)                               # J dot

            # print("M_d shape:", M_d.shape)
            # print("K_d shape:", K_d.shape)
            # print("B_d shape:", B_d.shape)
            # print("pos_error shape:", pos_error.shape)
            # print("vel_error shape:", vel_error.shape)


            #Acceleration
            x_dot_dot=np.linalg.inv(M_d) @ ((K_d @ pos_error ) + (B_d @ vel_error))
            #Joint Acceleration
            # print("jac_inv shape:", jac_inv.shape)
            # print("x_dot_dot shape:", x_dot_dot.shape)
            # print("jac_dot shape:", jac_dot.shape)
            # print("theta_dot shape:", theta_dot.shape)
            joint_acc=jac_inv @ ([0,0,0,x_dot_dot[0],x_dot_dot[1],x_dot_dot[2]]-(jac_dot @ theta_dot))
            print("I shape:", Mx.shape)
            # print("joint_acc shape:", joint_acc.shape)
            # print("H shape:", C.shape)
            tau=Mx @ joint_acc 
            # Jbar = M_inv @ jac.T @ Mx
            # ddq =  (q0 - data.qpos[dof_ids]) - data.qvel[dof_ids]
            # tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[:]=tau
            print(tau)
            mujoco.mj_step(model,data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__=="__main__":
    main()