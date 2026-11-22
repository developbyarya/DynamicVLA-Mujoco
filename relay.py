"""Carry DynamicVLA's two ZeroMQ streams over a WebSocket so the sim (your laptop) and the model
(a remote GPU, e.g. a Kaggle notebook) can live on different machines.

    laptop :  bridge.py <-ZMQ-> relay.py laptop  (WebSocket server)  <== Cloudflare tunnel ==
    remote :  inference.py <-ZMQ-> relay.py remote (WebSocket client) ==> the tunnel URL

Neither bridge.py nor scripts/inference.py is modified; each one talks to a local ZMQ socket as usual.
On the wire there is NO pickle: a JSON header plus JPEG images, and the remote must send the token
first. (Pickle over a public tunnel would let a stranger run code on your machine.)

Usage
    laptop: python relay.py laptop                       # prints the token to paste on the remote
    remote: python relay.py remote --url https://xxxx.trycloudflare.com --token TOKEN
Start order on the remote: relay first (it binds the ports), then scripts/inference.py.
"""
import argparse
import asyncio
import hmac
import io
import json
import logging
import secrets
import struct
import sys
import time

import numpy as np
import websockets
import zmq
import zmq.asyncio
from PIL import Image

log = logging.getLogger("relay")
_DTYPES = {"float32", "float64", "int32", "int64", "uint8", "bool"}


# ------------------------------------------------------------------ wire format
def _enc(o, blobs, q):
    if isinstance(o, dict):
        return {str(k): _enc(v, blobs, q) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(v, blobs, q) for v in o]
    if isinstance(o, np.ndarray):
        if o.dtype == np.uint8 and o.ndim == 4 and o.shape[0] == 1 and o.shape[3] == 3:
            buf = io.BytesIO()
            Image.fromarray(o[0]).save(buf, "JPEG", quality=q)
            blobs.append(buf.getvalue())
            return {"__jpeg__": len(blobs) - 1, "shape": list(o.shape)}
        return {"__nd__": o.tolist(), "dtype": str(o.dtype)}
    if isinstance(o, np.generic):
        return o.item()
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    raise TypeError("cannot send %s over the relay" % type(o))


def _dec(o, blobs):
    if isinstance(o, dict):
        if "__jpeg__" in o:
            im = np.asarray(Image.open(io.BytesIO(blobs[o["__jpeg__"]])).convert("RGB"))
            return im.reshape(o["shape"])
        if "__nd__" in o:
            if o["dtype"] not in _DTYPES:
                raise ValueError("bad dtype %r" % o["dtype"])
            return np.array(o["__nd__"], dtype=o["dtype"])
        return {k: _dec(v, blobs) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(v, blobs) for v in o]
    return o


def pack(msg, quality=90):
    blobs = []
    head = json.dumps({"m": _enc(msg, blobs, quality), "lens": [len(b) for b in blobs]}).encode()
    return struct.pack(">I", len(head)) + head + b"".join(blobs)


def unpack(data):
    (n,) = struct.unpack(">I", data[:4])
    head = json.loads(data[4:4 + n])
    off, blobs = 4 + n, []
    for length in head["lens"]:
        blobs.append(data[off:off + length])
        off += length
    return _dec(head["m"], blobs)


class Stats:
    def __init__(self, name):
        self.name, self.n, self.bytes, self.t = name, 0, 0, time.time()

    def add(self, nbytes):
        self.n += 1
        self.bytes += nbytes
        dt = time.time() - self.t
        if dt >= 5:
            log.info("%s: %.1f msg/s, %.0f KB/msg, %.2f MB/s", self.name, self.n / dt,
                     self.bytes / max(self.n, 1) / 1024, self.bytes / dt / 1e6)
            self.n, self.bytes, self.t = 0, 0, time.time()


async def _run_until_first_done(*coros):
    tasks = [asyncio.ensure_future(c) for c in coros]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        t.result()  # re-raise


# ------------------------------------------------------------------ laptop side (server)
async def run_laptop(a):
    ctx = zmq.asyncio.Context()
    obs = ctx.socket(zmq.SUB)  # from bridge.py
    obs.setsockopt(zmq.CONFLATE, 1)  # keep only the newest observation, like the model does
    obs.setsockopt_string(zmq.SUBSCRIBE, "")
    obs.connect("tcp://127.0.0.1:%d" % a.img_port)
    act = ctx.socket(zmq.PUSH)  # to bridge.py
    act.connect("tcp://127.0.0.1:%d" % a.act_port)
    token = a.token or secrets.token_urlsafe(16)

    async def handler(ws):
        try:
            first = json.loads(await asyncio.wait_for(ws.recv(), 10))
            ok = hmac.compare_digest(str(first.get("token", "")), token)
        except Exception:
            ok = False
        if not ok:
            log.warning("rejected a connection (bad token)")
            await ws.close(4401, "bad token")
            return
        log.info("remote model connected")
        s_up, s_down = Stats("obs  -> remote"), Stats("act  <- remote")

        async def up():
            while True:
                data = pack(await obs.recv_pyobj(), a.jpeg_quality)
                await ws.send(data)
                s_up.add(len(data))

        async def down():
            async for data in ws:
                if isinstance(data, bytes):
                    await act.send_pyobj(unpack(data))
                    s_down.add(len(data))

        try:
            await _run_until_first_done(up(), down())
        except websockets.ConnectionClosed:
            pass
        log.info("remote model disconnected")

    async with websockets.serve(handler, "127.0.0.1", a.port, max_size=None):
        print("\n=== relay ready on ws://127.0.0.1:%d ===" % a.port)
        print("1) in another terminal:  cloudflared tunnel --url http://localhost:%d" % a.port)
        print("2) on the remote:        RELAY_URL = the https://....trycloudflare.com it prints")
        print("                         RELAY_TOKEN = %s\n" % token, flush=True)
        await asyncio.Future()


# ------------------------------------------------------------------ remote side (client)
async def run_remote(a):
    ctx = zmq.asyncio.Context()
    pub = ctx.socket(zmq.PUB)  # inference.py subscribes here
    pub.bind("tcp://127.0.0.1:%d" % a.img_port)
    pull = ctx.socket(zmq.PULL)  # inference.py pushes handshake + actions here
    pull.bind("tcp://127.0.0.1:%d" % a.act_port)
    url = a.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    log.info("local ZMQ ports bound (%d/%d); connecting to %s", a.img_port, a.act_port, url)

    while True:
        try:
            async with websockets.connect(url, max_size=None, open_timeout=20, ping_interval=20) as ws:
                await ws.send(json.dumps({"token": a.token}))
                log.info("connected to the laptop")
                s_in, s_out = Stats("obs  <- laptop"), Stats("act  -> laptop")

                async def up():
                    async for data in ws:
                        if isinstance(data, bytes):
                            await pub.send_pyobj(unpack(data))
                            s_in.add(len(data))

                async def down():
                    while True:
                        data = pack(await pull.recv_pyobj())
                        await ws.send(data)
                        s_out.add(len(data))

                await _run_until_first_done(up(), down())
        except Exception as e:
            log.warning("connection problem (%s: %s); retrying in 3 s", type(e).__name__, e)
        await asyncio.sleep(3)


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="role", required=True)
    pl = sub.add_parser("laptop")
    pl.add_argument("--port", type=int, default=8765, help="WebSocket port for cloudflared")
    pl.add_argument("--token", default=None, help="default: random, printed at startup")
    pl.add_argument("--img_port", type=int, default=3186, help="bridge.py's observation port")
    pl.add_argument("--act_port", type=int, default=3188, help="bridge.py's action port")
    pl.add_argument("--jpeg_quality", type=int, default=90)
    pr = sub.add_parser("remote")
    pr.add_argument("--url", required=True)
    pr.add_argument("--token", required=True)
    pr.add_argument("--img_port", type=int, default=3186, help="local port inference.py subscribes to")
    pr.add_argument("--act_port", type=int, default=3188, help="local port inference.py pushes to")
    args = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_laptop(args) if args.role == "laptop" else run_remote(args))
