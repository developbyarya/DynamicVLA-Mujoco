"""Sensors/sites that make a MuJoCo Panda look like the DynamicVLA (DOM / Isaac Lab) setup.

Call add_dynamicvla_sensors(panda_spec) inside ConveyorEnv._build_model, in place of the
old hand.add_site(...) / hand.add_camera(...) lines.
"""

# Isaac cameras: focal 2.3 mm, aperture 4.6 mm -> 90 deg horizontal FOV. At 4:3 (480x360)
# MuJoCo's *vertical* fov is 2*atan(0.75) = 73.74 deg.
FOVY = 73.74


def add_dynamicvla_sensors(panda_spec):
    hand = panda_spec.body("hand")
    # State point: training reports the pose of panda_hand + [0, 0, 0.1034] (ee_frame).
    hand.add_site(name="ee", pos=[0, 0, 0.1034], size=[0.008] * 3, rgba=[1, 0, 0, 1])
    # Control point: training IK drives panda_hand + [0, 0, 0.107] to the commanded pose.
    hand.add_site(name="tcp", pos=[0, 0, 0.107], size=[0.006] * 3, rgba=[0, 1, 0, 1])
    # Wrist camera exactly as in training: pos [0.065,0,0], quat wxyz [0,.7071,.7071,0] (OpenGL
    # convention = MuJoCo camera convention: looks along -Z, +Y up).
    hand.add_camera(name="wrist_cam", pos=[0.065, 0, 0],
                    quat=[0, 0.7071068, 0.7071068, 0], fovy=FOVY)

    # Fixed cameras from simulations/configs/sim_cfg.yaml, expressed in the robot-base frame
    # (rotation given as Euler "XYZ" degrees there; converted to wxyz here).
    base = panda_spec.body("link0")
    base.add_camera(name="opst_cam", pos=[1.0, 0.0, 0.6],      # euler [0, 60, 90]
                    quat=[0.612372, 0.353553, 0.353553, 0.612372], fovy=FOVY)
    base.add_camera(name="side_cam", pos=[0.5, 1.0, 0.35],     # euler [-90, 0, 180]
                    quat=[0.0, 0.0, 0.707107, 0.707107], fovy=FOVY)
