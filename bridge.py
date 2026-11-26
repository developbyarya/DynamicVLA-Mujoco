"""MuJoCo conveyor <-> DynamicVLA bridge (same ZMQ protocol as simulations/evaluate.py).

Run this (sim side) first, then in the DynamicVLA repo:
    python3 scripts/inference.py -p /path/to/checkpoint -r euler -d -s
"""
import argparse
import logging
import time

import numpy as np
import zmq

from sim import ConveyorEnv, DEFAULT_SCANNED_ROOT, PRIMITIVE_KINDS, list_scanned_objects
from panda_ctrl import PandaController

DT = 0.04  # training control period (25 Hz)
CAMS = ["wrist_cam", "opst_cam"]  # the official checkpoint's input_features (no side_cam)
# Franka init pose from simulations/configs/sim_cfg.yaml (pos, quat wxyz) + gripper open flag
INIT_POSE = np.array([0.465906, 0.0, 0.382970, 0.008583, 0.921765, 0.020404, 0.387116, 1.0])


def main(a):
    if a.list_objects:
        print("Primitives:", ", ".join(PRIMITIVE_KINDS))
        scanned = list_scanned_objects(a.scanned_root)
        src = a.scanned_root or str(DEFAULT_SCANNED_ROOT)
        print(f"Scanned ({src}):")
        for name in scanned:
            print(f"  - {name}")
        return

    env = ConveyorEnv(belt_speed=a.belt_speed, seed=a.seed, scanned_root=a.scanned_root,
                      objects=a.objects, batch_size=a.batch_size,
                      collision=a.collision, object_scale=a.object_scale,
                      n_pool=a.n_pool, spawn_interval=a.spawn_interval)
    env.reset()
    env.next_spawn = 1e9  # no objects while the arm moves to the init pose
    ctrl = PandaController(env.model, env.data)
    substeps = int(round(DT / env.model.opt.timestep))

    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    def tick(action):
        if viewer is None:
            ctrl.apply(action)
            for _ in range(substeps):
                env.step()
        else:
            with viewer.lock():  # the viewer thread reads model/data; mutate only under the lock
                ctrl.apply(action)
                for _ in range(substeps):
                    env.step()
            viewer.sync()

    for _ in range(75):  # ~3 s to reach the init pose
        tick(INIT_POSE)
    env.next_spawn = env.data.time + a.spawn_delay

    zctx = zmq.Context()
    obs_sock = zctx.socket(zmq.PUB); obs_sock.bind("tcp://%s:%d" % (a.host, a.img_port))
    act_sock = zctx.socket(zmq.PULL); act_sock.bind("tcp://%s:%d" % (a.host, a.act_port))
    logging.info("Waiting for the DynamicVLA client on %s:%d / %d ...", a.host, a.img_port, a.act_port)
    while True:
        if act_sock.poll(1000):
            m = act_sock.recv_pyobj()
            if "vla" in m:
                logging.info("Client connected: %s", m)
                break
    time.sleep(1.0)  # let the SUB subscription reach the PUB, or the task message is lost
    obs_sock.send_pyobj({"task": a.task})  # also resets the model

    last = INIT_POSE[None].copy()
    idx, dt_scale, t0 = 0, 1.0, time.perf_counter()
    got_action = False
    while env.data.time < a.duration and (viewer is None or viewer.is_running()):
        obs = {
            "dt_scale": dt_scale,
            "index": idx,
            "observation.state": {"end_effector": {}},
        }
        pos, quat = ctrl.get_state()
        obs["observation.state"]["end_effector"] = {
            "pos": pos[None].astype(np.float32), "quat": quat[None].astype(np.float32)}
        for c in a.cams:
            obs["observation.images.%s" % c] = env.render(c, 480, 360)[None]  # uint8 (1,H,W,3)
        obs_sock.send_pyobj(obs)

        while act_sock.poll(0):  # keep only the newest action
            m = act_sock.recv_pyobj()
            if "action" in m:
                last = np.asarray(m["action"], float).reshape(-1, 8)[-1:]
                got_action = True

        tick(last[0])
        elapsed = time.perf_counter() - t0
        dt_scale = max(1.0, elapsed / DT)  # >1 when the sim is slower than real time
        if elapsed < DT and not a.no_realtime:
            time.sleep(DT - elapsed)
        t0 = time.perf_counter()
        idx += 1
        if idx % 25 == 0:
            logging.info("t=%.1fs step=%d actions_received=%s active_objects=%d ee=%s dt_scale=%.2f",
                         env.data.time, idx, got_action, int(env.active.sum()),
                         np.round(pos, 3), dt_scale)
    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick up the apple.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--img_port", type=int, default=3186)
    p.add_argument("--act_port", type=int, default=3188)
    p.add_argument("--duration", type=float, default=30.0, help="simulated seconds")
    p.add_argument("--belt_speed", type=float, default=0.05)
    p.add_argument("--spawn_interval", type=float, default=3.0)
    p.add_argument("--spawn_delay", type=float, default=1.0)
    p.add_argument("--n_pool", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cams", nargs="+", default=CAMS, help="cameras to stream; must cover the checkpoint's observation.images.* keys")
    p.add_argument("--scanned_root", "--scanned-root", "--scanned", dest="scanned_root", default=None)
    p.add_argument("--object", "--objects", dest="objects", nargs="+", default=None,
                   help="object(s) to spawn: primitive(s) (box, sphere, cylinder, capsule) or scanned model name/substring")
    p.add_argument("--list-objects", "--list_objects", dest="list_objects", action="store_true",
                   help="list available primitive and scanned objects and exit")
    p.add_argument("--batch_size", "--batch-size", dest="batch_size", default="1,3",
                   help="objects spawned per batch: 'N' or 'MIN,MAX' (default: 1,3)")
    p.add_argument("--object_scale", "--object-scale", dest="object_scale", type=float, default=1.0,
                   help="scale multiplier for spawned objects (e.g. 0.65 so 10 cm cups fit 8 cm gripper)")
    p.add_argument("--collision", choices=["hull", "submeshes"], default="hull")
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer window")
    p.add_argument("--no_realtime", action="store_true", help="debug: don't sleep to 25 Hz wall-clock")
    main(p.parse_args())
