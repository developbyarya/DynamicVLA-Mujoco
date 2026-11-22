"""
Subscriber ZeroMQ (SUB) untuk menerima dan menampilkan stream kamera wrist_cam dari sim.py.

Pemakaian:
  python sub_cam.py
  python sub_cam.py --addr tcp://127.0.0.1:5555 --topic wrist_cam
  python sub_cam.py --backend tk
"""
import argparse
import json
import struct
import time

import zmq


def decode_frame(payload, topic_prefix):
    """Parse payload [topic ' '][4B meta_len][meta_json][rgb_bytes] -> (meta, rgb_bytes)."""
    body = payload[len(topic_prefix):]
    (meta_len,) = struct.unpack("!I", body[:4])
    meta_end = 4 + meta_len
    meta = json.loads(body[4:meta_end].decode("utf-8"))
    rgb_bytes = body[meta_end:]
    return meta, rgb_bytes


def create_sub_socket(addr, topic):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    prefix = f"{topic} ".encode("utf-8")
    sock.setsockopt(zmq.SUBSCRIBE, prefix)
    sock.connect(addr)
    return sock, prefix


def run_cv2(sock, prefix):
    import cv2
    import numpy as np

    win = "wrist_cam"
    last_t = time.monotonic()
    fps = 0.0
    try:
        while True:
            if sock.poll(timeout=50):
                payload = sock.recv()
                meta, rgb_bytes = decode_frame(payload, prefix)
                w, h = meta["width"], meta["height"]
                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                now = time.monotonic()
                dt = now - last_t
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                last_t = now

                label = f"t={meta['sim_time']:.2f}s  frame={meta['frame']}  {fps:.1f} fps"
                cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(win, bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


def run_tk(sock, prefix):
    import tkinter as tk

    root = tk.Tk()
    root.title("wrist_cam (menunggu frame...)")
    label = tk.Label(root)
    label.pack()

    state = {"photo": None, "last_t": time.monotonic(), "fps": 0.0}

    def poll_frame():
        try:
            payload = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            payload = None

        if payload is not None:
            meta, rgb_bytes = decode_frame(payload, prefix)
            w, h = meta["width"], meta["height"]
            ppm = f"P6 {w} {h} 255\n".encode("ascii") + rgb_bytes
            photo = tk.PhotoImage(data=ppm, format="PPM")
            state["photo"] = photo
            label.configure(image=photo)

            now = time.monotonic()
            dt = now - state["last_t"]
            if dt > 0:
                state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / dt) if state["fps"] > 0 else (1.0 / dt)
            state["last_t"] = now
            root.title(f"wrist_cam | t={meta['sim_time']:.2f}s | frame={meta['frame']} | {state['fps']:.1f} fps")

        root.after(10, poll_frame)

    root.bind("<q>", lambda _e: root.destroy())
    root.bind("<Escape>", lambda _e: root.destroy())
    root.after(10, poll_frame)
    root.mainloop()


def select_backend(requested):
    if requested != "auto":
        return requested
    try:
        import cv2  # noqa: F401
        return "cv2"
    except ImportError:
        return "tk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://127.0.0.1:5555", help="endpoint ZMQ PUB yang dituju")
    ap.add_argument("--topic", default="wrist_cam", help="nama topik ZMQ SUB")
    ap.add_argument("--backend", choices=["auto", "cv2", "tk"], default="auto", help="backend tampilan")
    args = ap.parse_args()

    backend = select_backend(args.backend)
    sock, prefix = create_sub_socket(args.addr, args.topic)
    print(f"Subscribed to {args.addr} topic='{args.topic}' (backend={backend})")
    try:
        if backend == "cv2":
            run_cv2(sock, prefix)
        else:
            run_tk(sock, prefix)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
