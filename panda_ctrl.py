import numpy as np
import mujoco

N_ARM = 7


class PandaController:
    """DynamicVLA action [x,y,z,qw,qx,qy,qz,grip] -> MuJoCo Panda joint targets.

    - Poses are in the robot-base (link0) frame, so the arm can be mounted/rotated anywhere.
    - get_state() reads the "ee" site (what training reports); apply() drives the "tcp" site
      (what training's IK controlled). They differ by 3.6 mm along the tool z axis, as in training.
    """

    def __init__(self, model, data, ctrl_site="tcp", state_site="ee", base_body="link0",
                 damping=0.05, max_dq=0.2):
        self.m, self.d = model, data
        self.tcp = model.site(ctrl_site).id
        self.ee = model.site(state_site).id
        self.base = model.body(base_body).id
        self.hand = model.body("hand").id
        self.lam, self.max_dq = damping, max_dq
        # Look arm joints/actuators up by name: in a scene with other free bodies they are NOT
        # at qpos[0:7] / dof[0:7] / ctrl[0:7].
        jnts = [model.joint("joint%d" % i) for i in range(1, N_ARM + 1)]
        self.qadr = np.array([int(model.jnt_qposadr[j.id]) for j in jnts])
        self.dadr = np.array([int(model.jnt_dofadr[j.id]) for j in jnts])
        self.act = np.array([model.actuator("actuator%d" % i).id for i in range(1, N_ARM + 1)])
        self.grip = model.actuator("actuator8").id
        self.lo = model.actuator_ctrlrange[self.act, 0]
        self.hi = model.actuator_ctrlrange[self.act, 1]

    # ---- frames ----
    def _base(self):
        return self.d.xpos[self.base].copy(), self.d.xquat[self.base].copy()

    def _to_world(self, pos_b, quat_b):
        bp, bq = self._base()
        pw = np.zeros(3); qw = np.zeros(4)
        mujoco.mju_rotVecQuat(pw, pos_b, bq)
        mujoco.mju_mulQuat(qw, bq, quat_b)
        return pw + bp, qw

    def _site_pose_world(self, sid):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[sid])
        return self.d.site_xpos[sid].copy(), q

    def get_state(self):
        """TCP pose in the base frame: (pos (3,), quat wxyz (4,))."""
        bp, bq = self._base()
        pw, qw = self._site_pose_world(self.ee)
        inv = np.zeros(4); mujoco.mju_negQuat(inv, bq)
        pb = np.zeros(3); qb = np.zeros(4)
        mujoco.mju_rotVecQuat(pb, pw - bp, inv)
        mujoco.mju_mulQuat(qb, inv, qw)
        return pb, qb

    # ---- one control step (like Isaac's DifferentialIK: absolute pose, DLS, single step) ----
    def apply(self, action):
        a = np.asarray(action, float).reshape(-1)
        pos_t, quat_t = self._to_world(a[:3], a[3:7] / np.linalg.norm(a[3:7]))
        pos_c, quat_c = self._site_pose_world(self.tcp)

        e = np.zeros(6)
        e[:3] = pos_t - pos_c
        qc_inv = np.zeros(4); mujoco.mju_negQuat(qc_inv, quat_c)
        dq = np.zeros(4); mujoco.mju_mulQuat(dq, quat_t, qc_inv)
        if dq[0] < 0:
            dq = -dq
        mujoco.mju_quat2Vel(e[3:], dq, 1.0)

        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.tcp)
        J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
        dq_j = J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(6), e)
        dq_j = np.clip(dq_j, -self.max_dq, self.max_dq)

        self.d.ctrl[self.act] = np.clip(self.d.qpos[self.qadr] + dq_j, self.lo, self.hi)
        # gripper flag: >= 0 open, < 0 close (assumed Isaac BinaryJointPositionAction convention)
        self.d.ctrl[self.grip] = 255.0 if a[7] >= 0 else 0.0
        return e
