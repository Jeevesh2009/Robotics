import mujoco
import mujoco.viewer
import time
import numpy as np

#--------------------------------------------------Impedance Control---------------------------------------------------


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
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
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


       

        M_d=0.0015*np.eye(3)
        K_d=50*np.eye(3)
        B_d=100*np.eye(3)
        while viewer.is_running():
            step_start = time.time()

             #Desired Position
            x_d=data.mocap_pos[mocap_id]


            #Desired Velocity
            v_d=np.zeros(3)
            #joint positions and velocities
            theta = data.qpos[dof_ids]
            theta_dot=data.qvel[dof_ids]

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac_temp[:3], jac_temp[3:], site_id)
            jac=jac_temp[:,[0,1,2,3,4,5,6]]

            # CS positions and velocities
            position=data.site(site_id).xpos.copy()
            velocity=jac[:3] @ theta_dot
            


            #Errors in pos and velocity
            pos_error = x_d - position
            vel_error = v_d - velocity

            F = B_d @ vel_error + K_d @ pos_error
            print(F.shape)
            tau= jac.T @ (F[0],F[1],F[2],1,0,0)
            # Jbar = M_inv @ jac.T @ Mx
            ddq =  (q0 - data.qpos[dof_ids]) - data.qvel[dof_ids]
            tau += (np.eye(model.nv) - jac.T @ np.linalg.pinv(jac).T) @ ddq
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