"""Stand-in for scripts/inference.py: same sockets/handshake, but a scripted policy (no model).
Used only to test the sim side of the bridge."""
import sys
import time

import numpy as np
import zmq
from PIL import Image

ctx = zmq.Context()
obs = ctx.socket(zmq.SUB); obs.connect("tcp://127.0.0.1:3186"); obs.setsockopt_string(zmq.SUBSCRIBE, ""); obs.RCVHWM = 1
act = ctx.socket(zmq.PUSH); act.connect("tcp://127.0.0.1:3188")
time.sleep(0.5)
act.send_pyobj({"vla": "fake", "epoch": 0})

init = np.array([0.465906, 0.0, 0.382970, 0.008583, 0.921765, 0.020404, 0.387116, 1.0])
n, task, saved, log = 0, None, False, []
t_end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 12
while time.time() < t_end:
    if not obs.poll(500):
        continue
    o = obs.recv_pyobj()
    if "task" in o:
        task = o["task"]; print("task:", task); continue
    n += 1
    st = o["observation.state"]["end_effector"]
    if not saved:
        for k, v in o.items():
            if k.startswith("observation.images"):
                print(k, v.shape, v.dtype)
                Image.fromarray(v[0]).save("/tmp/claude-0/-home-claude/47a0d4e0-105b-5b71-946e-3148f6349569/scratchpad/bridge/%s.png" % k.split(".")[-1])
        print("state:", {k: np.round(v, 3) for k, v in st.items()}, "index", o["index"], "dt_scale", o["dt_scale"])
        saved = True
    # scripted motion: move +10cm in x/-10cm in y over 3 s, then close gripper at 4 s
    a = init.copy()
    s = min(1.0, n / 75)
    a[:3] += s * np.array([0.10, -0.10, -0.05])
    a[7] = -1.0 if n > 100 else 1.0
    act.send_pyobj({"action": a[None]})
    log.append((n, st["pos"][0].round(3).tolist(), a[:3].round(3).tolist()))
print("last steps (n, measured pos, commanded pos):", log[-1], "| total obs", n)
