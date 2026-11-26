"""
Environment konveyor + lengan robot paling sederhana di MuJoCo.

Konsep:
  - Sabuk konveyor lurus (sumbu +X). Objek bergerak dari ujung kiri ke ujung kanan.
  - Objek muncul (spawn) per "batch" yang menumpuk/berdempetan -> cluttered.
  - Objek yang mencapai ujung sabuk dihilangkan (despawn) lalu dipakai ulang dari pool.
  - Pergerakan sabuk disimulasikan secara kinematik: kecepatan sumbu-X objek ditarik
    menuju belt_speed (MuJoCo tidak punya permukaan bergerak). Tumbukan antar objek,
    gesekan, dan gravitasi tetap dihitung fisika, jadi tumpukan tetap saling bertabrakan.
  - Lengan robot Franka Emika Panda (7-DOF + gripper 2 jari Franka Hand) dipasang di sisi
    sabuk, tepat di tengah panjang sabuk (x = 0). Gravitasi lengan dikompensasi, kontrol
    memakai target posisi sendi.

Pemakaian:
  pip install mujoco numpy
  python conveyor_env.py                            # viewer interaktif, lengan diam
  python conveyor_env.py --demo                     # lengan bergerak mengikuti sabuk
  python conveyor_env.py --headless --duration 30   # tanpa tampilan, cetak statistik
  python conveyor_env.py --demo --scanned path/to/mujoco_scanned_objects   # objek scan Google
  (tambahkan --collision submeshes untuk collision 32 bagian, --names A,B,C untuk memilih objek)

Untuk riset lanjutan, pakai kelas ConveyorEnv langsung:
  env = ConveyorEnv(); env.reset()
  while True:
      env.set_arm_targets(q7)      # 7 sudut sendi (rad): joint1..joint7
      env.set_gripper(closed=True) # buka/tutup gripper
      info = env.step()            # info["positions"] = posisi objek aktif (N, 3), info["ee_pos"] = ujung gripper
"""
import argparse
import json
import struct
import time
from pathlib import Path
from dynamicvla_cams import add_dynamicvla_sensors

import numpy as np
import mujoco

PANDA_XML = Path(__file__).resolve().parent / "franka_emika_panda" / "panda.xml"
DEFAULT_SCANNED_ROOT = Path(__file__).resolve().parent / "scanned_objects_mini"
PRIMITIVE_KINDS = ("box", "sphere", "cylinder", "capsule")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
# joint1..joint7 -> gripper menghadap ke bawah di atas tengah sabuk
ARM_HOME = np.array([0.0, -0.05, 0.0, -2.25, 0.0, 2.20, np.pi / 4])


def parse_object_args(values):
    """Parse daftar nama objek dari string dipisah koma atau list CLI args."""
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    out = []
    for item in values:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def parse_batch_size(value):
    """Parse --batch-size seperti '1' -> (1, 1) atau '3,6' -> (3, 6)."""
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            n = int(value[0])
            return (n, n)
        return (int(value[0]), int(value[1]))
    parts = [int(p.strip()) for p in str(value).split(",") if p.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2 and 1 <= parts[0] <= parts[1]:
        return (parts[0], parts[1])
    raise ValueError(f"Format batch_size tidak valid: {value!r} (gunakan 'N' atau 'MIN,MAX')")


def list_scanned_objects(scanned_root=None):
    """Daftar nama folder model yang tersedia di direktori scanned objects."""
    root = Path(scanned_root) if scanned_root else DEFAULT_SCANNED_ROOT
    models = root / "models"
    if not models.is_dir():
        return []
    return sorted(p.name for p in models.iterdir() if p.is_dir() and (p / "model.obj").exists())


class ConveyorEnv:
    def __init__(
        self,
        belt_length=1.0,
        belt_width=0.5,
        belt_speed=0.05,        # m/s
        belt_gain=0.2,          # seberapa kuat objek ditarik ke kecepatan sabuk per step
        n_pool=3,              # jumlah maksimum objek yang ada sekaligus
        spawn_interval=3.0,     # detik antar batch
        batch_size=(1, 3),      # jumlah objek per batch (min, maks)
        seed=0,
        scanned_root=None,      # path ke clone mujoco_scanned_objects -> pakai objek scan, bukan primitif
        scanned_names=None,     # daftar nama folder objek; None = ambil acak dari yang ada
        objects=None,           # pilih objek spawn: primitif (box/sphere/cylinder/capsule) atau nama/substring folder scan
        n_distinct=15,          # jumlah jenis objek scan berbeda di pool
        max_extent=0.12,        # buang objek yang sisi terpanjangnya > ini (meter), agar muat di gripper
        collision="hull",       # "hull" = 1 convex hull per objek (cepat); "submeshes" = 32 bagian V-HACD (akurat)
        object_mass=0.1,        # massa tiap objek scan (kg)
        object_scale=1.0,       # faktor skala ukuran objek (mis. 0.65 agar cangkir 10 cm muat di gripper 8 cm)
    ):
        self.belt_length = belt_length
        self.belt_width = belt_width
        self.belt_speed = belt_speed
        self.belt_gain = belt_gain
        self.n_pool = n_pool
        self.spawn_interval = spawn_interval
        self.batch_size = parse_batch_size(batch_size)
        self.start_x = -belt_length / 2 + 0.15
        self.end_x = belt_length / 2          # objek dihapus saat pusatnya melewati ini
        self.arm_base_y = -belt_width / 2 - 0.2   # lengan di sisi sabuk, x = 0 (tengah)
        self.rng = np.random.default_rng(seed)
        self.collision = collision
        self.object_mass = object_mass
        self.object_scale = float(object_scale)
        self.primitive_kinds = list(PRIMITIVE_KINDS)

        selected = parse_object_args(objects) or parse_object_args(scanned_names)
        if scanned_root is not None:
            self.scanned = self._find_scanned(scanned_root, selected, n_distinct, max_extent)
        elif selected is not None:
            lowered = [s.lower() for s in selected]
            if all(s in PRIMITIVE_KINDS for s in lowered):
                self.primitive_kinds = lowered
                self.scanned = None
            else:
                if (DEFAULT_SCANNED_ROOT / "models").is_dir():
                    self.scanned = self._find_scanned(DEFAULT_SCANNED_ROOT, selected, n_distinct, max_extent)
                else:
                    raise ValueError(
                        f"Objek tidak dikenal: {selected}. Primitif valid: {list(PRIMITIVE_KINDS)}, "
                        f"atau berikan --scanned_root / --scanned."
                    )
        else:
            self.scanned = None

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}") for i in range(n_pool)]
        # semua geom milik tiap objek (1 untuk primitif; 1 visual + collision untuk objek scan)
        self.obj_geoms = [np.arange(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]) for b in self.body_ids]
        jnt_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_j") for i in range(n_pool)]
        self.qadr = [int(m.jnt_qposadr[j]) for j in jnt_ids]
        self.vadr = [int(m.jnt_dofadr[j]) for j in jnt_ids]
        self.ct0 = m.geom_contype.copy()
        self.ca0 = m.geom_conaffinity.copy()
        self.rgba0 = m.geom_rgba.copy()
        self.obj_radius = [float(m.geom_rbound[g].max()) for g in self.obj_geoms]   # untuk tinggi spawn

        # lengan robot (Franka Emika Panda)
        arm_jnt = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qadr = [int(m.jnt_qposadr[j]) for j in arm_jnt]
        self.arm_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{k}") for k in range(1, 8)]
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.wrist_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self._renderer = None
        self._renderer_size = None
        self._render_opt = mujoco.MjvOption()
        self._render_opt.sitegroup[:] = 0

        self.active = np.zeros(n_pool, dtype=bool)
        self.n_spawned = 0
        self.n_despawned = 0
        self.next_spawn = 0.0

    # ------------------------------------------------------------------ model
    def _find_scanned(self, root, names, n_distinct, max_extent):
        """Pilih folder objek dari mujoco_scanned_objects yang cukup kecil untuk gripper."""
        models = Path(root) / "models"
        available = sorted(p.name for p in models.iterdir() if p.is_dir() and (p / "model.obj").exists())
        explicit = names is not None
        if not explicit:
            names = [available[k] for k in self.rng.permutation(len(available))]
        else:
            resolved = []
            for query in names:
                if query in available:
                    matches = [query]
                else:
                    q_low = query.lower()
                    matches = [a for a in available if q_low in a.lower()]
                if not matches:
                    raise ValueError(
                        f"Objek {query!r} tidak ditemukan di {models}. "
                        f"Tersedia: {available} (primitif: {list(PRIMITIVE_KINDS)})"
                    )
                for m in matches:
                    if m not in resolved:
                        resolved.append(m)
            names = resolved

        found = []
        for n in names:
            obj = models / n / "model.obj"
            if not obj.exists():
                continue
            if not explicit:
                verts = np.array([[float(x) for x in ln.split()[1:4]]
                                  for ln in obj.read_text().splitlines() if ln.startswith("v ")])
                if np.ptp(verts, axis=0).max() * self.object_scale > max_extent:
                    continue
            found.append(models / n)
            if not explicit and len(found) >= n_distinct:
                break
        if not found:
            raise ValueError(f"Tidak ada objek <= {max_extent} m di {models}")
        return found

    def _build_model(self):
        spec = mujoco.MjSpec.from_string(self._build_xml())
        panda_spec = mujoco.MjSpec.from_file(str(PANDA_XML))

        # Kompensasi gravitasi untuk semua body Panda agar kontrol posisi tidak melorot
        for body in panda_spec.bodies:
            body.gravcomp = 1

        # Tambahkan site end-effector dan wrist_cam pada Franka Hand
        add_dynamicvla_sensors(panda_spec)

        mount = spec.worldbody.add_site(
            name="arm_mount",
            pos=[0, self.arm_base_y, 0],
            quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],  # rotasi +90 deg Z menghadap sabuk (+Y)
        )
        spec.attach(panda_spec, prefix="", suffix="", site=mount)
        return spec.compile()

    def _build_xml(self):
        L, W = self.belt_length, self.belt_width
        objs, assets, ncoll = [], [], {}
        scale_attr = (
            f' scale="{self.object_scale:.6f} {self.object_scale:.6f} {self.object_scale:.6f}"'
            if self.object_scale != 1.0
            else ""
        )
        if self.scanned:
            for j, d in enumerate(self.scanned):
                p = d.as_posix()
                assets.append(f'<texture type="2d" name="o{j}_tex" file="{p}/texture.png"/>'
                              f'<material name="o{j}_mat" texture="o{j}_tex" specular="0.5" shininess="0.5"/>'
                              f'<mesh name="o{j}_vis" file="{p}/model.obj"{scale_attr}/>')
                if self.collision == "submeshes":
                    files = sorted(d.glob("model_collision_*.obj"))
                    ncoll[j] = len(files)
                    for k, f in enumerate(files):
                        assets.append(f'<mesh name="o{j}_c{k}" file="{f.as_posix()}"{scale_attr}/>')
        for i in range(self.n_pool):
            if self.scanned:
                j = i % len(self.scanned)
                cols = [f"o{j}_vis"] if self.collision == "hull" else [f"o{j}_c{k}" for k in range(ncoll[j])]
                geoms = (f'<geom name="obj{i}_g" type="mesh" mesh="o{j}_vis" material="o{j}_mat" '
                         f'contype="0" conaffinity="0" group="2" mass="0"/>')   # visual saja, tanpa massa
                geoms += "".join(
                    f'<geom type="mesh" mesh="{c}" group="3" mass="{self.object_mass / len(cols):.6f}" '
                    f'friction="0.8 0.01 0.001"/>' for c in cols)
                objs.append(f'<body name="obj{i}" pos="0 3 -1"><freejoint name="obj{i}_j"/>{geoms}</body>')
                continue
            s = self.rng.uniform(0.02, 0.035) * self.object_scale
            kind = self.primitive_kinds[i % len(self.primitive_kinds)]
            if kind == "box":
                size = " ".join(f"{v:.4f}" for v in s * self.rng.uniform(0.7, 1.3, 3))
            elif kind == "sphere":
                size = f"{s:.4f}"
            else:  # cylinder / capsule: radius, half-height
                size = f"{s:.4f} {s * self.rng.uniform(0.8, 1.5):.4f}"
            rgb = " ".join(f"{v:.2f}" for v in self.rng.uniform(0.2, 1.0, 3))
            objs.append(
                f'<body name="obj{i}" pos="0 3 -1">'
                f'<freejoint name="obj{i}_j"/>'
                f'<geom name="obj{i}_g" type="{kind}" size="{size}" rgba="{rgb} 1" '
                f'friction="0.8 0.01 0.001" mass="0.05"/>'
                f"</body>"
            )
        return f"""
<mujoco model="conveyor">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 -0.3" size="3 3 0.1" rgba="0.85 0.85 0.85 1"/>
    <geom name="belt" type="box" pos="0 0 -0.02" size="{L/2} {W/2} 0.02" rgba="0.25 0.25 0.28 1"
          friction="1.0 0.01 0.001"/>
    <geom name="rail_l" type="box" pos="0 {W/2 + 0.01} 0.03" size="{L/2} 0.01 0.05" rgba="0.6 0.6 0.6 1"/>
    <geom name="rail_r" type="box" pos="0 {-W/2 - 0.01} 0.03" size="{L/2} 0.01 0.05" rgba="0.6 0.6 0.6 1"/>
    <geom name="pedestal" type="cylinder" pos="0 {self.arm_base_y} -0.15" size="0.07 0.15" rgba="0.3 0.3 0.35 1"/>
    {''.join(objs)}
  </worldbody>
</mujoco>"""

    # ---------------------------------------------------------- spawn/despawn
    def _deactivate(self, i):
        gs = self.obj_geoms[i]
        self.model.geom_contype[gs] = 0
        self.model.geom_conaffinity[gs] = 0
        self.model.geom_rgba[gs, 3] = 0.0           # tidak terlihat
        q, v = self.qadr[i], self.vadr[i]
        self.data.qpos[q:q + 7] = [0.0, 3.0 + 0.1 * i, -1.0, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = 0.0
        self.active[i] = False

    def _activate(self, i, pos):
        gs = self.obj_geoms[i]
        self.model.geom_contype[gs] = self.ct0[gs]
        self.model.geom_conaffinity[gs] = self.ca0[gs]
        self.model.geom_rgba[gs] = self.rgba0[gs]
        quat = self.rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q, v = self.qadr[i], self.vadr[i]
        self.data.qpos[q:q + 7] = np.concatenate([pos, quat])
        self.data.qvel[v:v + 6] = 0.0
        self.active[i] = True

    def _spawn_batch(self):
        free = np.flatnonzero(~self.active)
        lo, hi = self.batch_size
        n = min(int(self.rng.integers(lo, hi + 1)), len(free))
        for k, i in enumerate(free[:n]):
            # posisi berdekatan + ditumpuk sedikit -> jatuh dan membentuk gerombolan
            pos = np.array([
                self.start_x + self.rng.uniform(-0.05, 0.05),
                self.rng.uniform(-0.12, 0.12),
                0.01 + self.obj_radius[i] + 0.09 * (k // 2),
            ])
            self._activate(i, pos)
            self.n_spawned += 1

    # -------------------------------------------------------------------- API
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.n_pool):
            self._deactivate(i)
        self.n_spawned = self.n_despawned = 0
        self.next_spawn = 0.0
        for adr, q in zip(self.arm_qadr, ARM_HOME):
            self.data.qpos[adr] = q
        self.set_arm_targets(ARM_HOME)
        self.set_gripper(closed=False)
        mujoco.mj_forward(self.model, self.data)
        return self.get_positions()

    def set_arm_targets(self, q):
        """Target posisi 7 sendi lengan Panda (rad)."""
        self.data.ctrl[self.arm_act] = q

    def set_gripper(self, closed):
        self.data.ctrl[self.grip_act] = 0.0 if closed else 255.0

    def ee_pos(self):
        """Posisi ujung gripper (end-effector) di koordinat dunia."""
        return self.data.site_xpos[self.ee_site].copy()

    def get_positions(self):
        idx = np.flatnonzero(self.active)
        return np.array([self.data.qpos[self.qadr[i]:self.qadr[i] + 3] for i in idx]).reshape(-1, 3)

    def render(self, camera="wrist_cam", width=320, height=240):
        """Render gambar RGB (H, W, 3) uint8 dari kamera (default: wrist_cam)."""
        if self._renderer is None or self._renderer_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._render_opt)
        return self._renderer.render()

    def get_wrist_image(self, width=320, height=240):
        """Ambil gambar RGB egosentris (H, W, 3) uint8 dari kamera wrist_cam."""
        return self.render(camera="wrist_cam", width=width, height=height)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_size = None

    def step(self):
        d = self.data
        if d.time >= self.next_spawn:
            self._spawn_batch()
            self.next_spawn += self.spawn_interval

        for i in range(self.n_pool):
            q, v = self.qadr[i], self.vadr[i]
            if not self.active[i]:
                # tahan objek yang sedang "disimpan" agar tidak jatuh terus
                d.qpos[q:q + 3] = [0.0, 3.0 + 0.1 * i, -1.0]
                d.qvel[v:v + 6] = 0.0
                continue
            x, y, z = d.qpos[q:q + 3]
            if x > self.end_x or z < -0.2 or abs(y) > self.belt_width:   # sampai ujung / jatuh
                self._deactivate(i)
                self.n_despawned += 1
                continue
            d.qvel[v] += self.belt_gain * (self.belt_speed - d.qvel[v])  # efek sabuk

        mujoco.mj_step(self.model, d)
        return {
            "time": d.time,
            "n_active": int(self.active.sum()),
            "n_spawned": self.n_spawned,
            "n_despawned": self.n_despawned,
            "positions": self.get_positions(),
            "ee_pos": self.ee_pos(),
        }


def demo_targets(t):
    """Gerakan contoh: lengan menyapu sepanjang sabuk, gripper buka-tutup tiap 2 detik."""
    q = ARM_HOME.copy()
    q[0] += 0.5 * np.sin(2 * np.pi * t / 8.0)    # joint1 (yaw) -> ujung gripper bergeser di sumbu X
    q[1] += 0.12 * np.sin(2 * np.pi * t / 4.0)   # sedikit maju-mundur / naik-turun
    q[3] -= 0.15 * np.sin(2 * np.pi * t / 4.0)
    return q, int(t // 2) % 2 == 1


class WristCamPublisher:
    """Publisher ZeroMQ (PUB) untuk streaming frame wrist_cam sebagai single-part message."""

    def __init__(self, addr="tcp://*:5555", topic="wrist_cam", fps=30.0, width=320, height=240):
        import zmq

        self.topic_prefix = f"{topic} ".encode("utf-8")
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.width = width
        self.height = height
        self.frame_id = 0
        self._last_pub = 0.0

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(addr)

    def publish_frame(self, img, sim_time):
        """Kirim frame RGB (H, W, 3) uint8 dengan format [topic ' '][4B len][json][rgb]."""
        h, w = img.shape[:2]
        meta = json.dumps({
            "width": int(w),
            "height": int(h),
            "channels": 3,
            "dtype": "uint8",
            "sim_time": float(sim_time),
            "frame": self.frame_id,
        }).encode("utf-8")
        payload = self.topic_prefix + struct.pack("!I", len(meta)) + meta + np.ascontiguousarray(img).tobytes()
        self.sock.send(payload)
        self.frame_id += 1

    def maybe_publish(self, env):
        """Render dan kirim frame jika interval target FPS sudah tercapai."""
        now = time.monotonic()
        if self.period > 0.0 and (now - self._last_pub) < self.period:
            return False
        self._last_pub = now
        img = env.get_wrist_image(width=self.width, height=self.height)
        self.publish_frame(img, env.data.time)
        return True

    def close(self):
        self.sock.close(linger=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="tanpa viewer, cetak statistik")
    ap.add_argument("--demo", action="store_true", help="gerakkan lengan secara otomatis")
    ap.add_argument("--duration", type=float, default=30.0, help="detik simulasi (headless)")
    ap.add_argument("--speed", type=float, default=0.25, help="kecepatan sabuk (m/s)")
    ap.add_argument("--interval", type=float, default=3.0, help="detik antar batch spawn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scanned", "--scanned-root", "--scanned_root", dest="scanned", default=None,
                    help="path ke clone mujoco_scanned_objects")
    ap.add_argument("--object", "--objects", dest="objects", nargs="+", default=None,
                    help="pilih objek untuk di-spawn: primitif (box, sphere, cylinder, capsule) atau nama/substring model scan")
    ap.add_argument("--names", default=None, help="alias lama untuk --object (dipisah koma)")
    ap.add_argument("--list-objects", "--list_objects", dest="list_objects", action="store_true",
                    help="cetak daftar objek primitif dan model scan yang tersedia lalu keluar")
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", default="3,6",
                    help="jumlah objek per batch: 'N' atau 'MIN,MAX' (default: 3,6)")
    ap.add_argument("--object-scale", "--object_scale", dest="object_scale", type=float, default=1.0,
                    help="faktor skala ukuran objek (default: 1.0)")
    ap.add_argument("--collision", choices=["hull", "submeshes"], default="hull")
    ap.add_argument("--max-extent", type=float, default=0.12, help="ukuran maks objek scan (m)")
    ap.add_argument("--wrist-cam", action="store_true", help="mulai viewer dari sudut pandang wrist_cam")
    ap.add_argument("--stream", action="store_true", help="stream wrist_cam lewat ZeroMQ PUB")
    ap.add_argument("--stream-addr", default="tcp://*:5555", help="endpoint bind ZMQ PUB")
    ap.add_argument("--stream-topic", default="wrist_cam", help="nama topik ZMQ PUB")
    ap.add_argument("--stream-fps", type=float, default=30.0, help="target FPS streaming")
    ap.add_argument("--stream-size", default="320x240", help="resolusi streaming WxH")
    args = ap.parse_args()

    if args.list_objects:
        print("Primitives:", ", ".join(PRIMITIVE_KINDS))
        scanned = list_scanned_objects(args.scanned)
        src = args.scanned or str(DEFAULT_SCANNED_ROOT)
        print(f"Scanned ({src}):")
        for name in scanned:
            print(f"  - {name}")
        return

    w_str, h_str = args.stream_size.lower().split("x", 1)
    stream_w, stream_h = int(w_str), int(h_str)

    env = ConveyorEnv(belt_speed=args.speed, spawn_interval=args.interval, seed=args.seed,
                      batch_size=args.batch_size,
                      scanned_root=args.scanned,
                      scanned_names=args.names,
                      objects=args.objects,
                      collision=args.collision, max_extent=args.max_extent,
                      object_scale=args.object_scale)
    env.reset()

    publisher = (
        WristCamPublisher(
            addr=args.stream_addr,
            topic=args.stream_topic,
            fps=args.stream_fps,
            width=stream_w,
            height=stream_h,
        )
        if args.stream
        else None
    )

    def control():
        if args.demo:
            q, closed = demo_targets(env.data.time)
            env.set_arm_targets(q)
            env.set_gripper(closed)

    try:
        if args.headless:
            next_print = 0.0
            while env.data.time < args.duration:
                control()
                info = env.step()
                if publisher is not None:
                    publisher.maybe_publish(env)
                if info["time"] >= next_print:
                    ee = info["ee_pos"]
                    print(f"t={info['time']:5.1f}s  aktif={info['n_active']:2d}  "
                          f"spawn={info['n_spawned']:3d}  despawn={info['n_despawned']:3d}  "
                          f"ee=({ee[0]:+.2f}, {ee[1]:+.2f}, {ee[2]:+.2f})")
                    next_print += 2.0
            return

        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            if args.wrist_cam:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = env.wrist_cam_id
            else:
                viewer.cam.lookat[:] = [0, -0.15, 0.1]
                viewer.cam.distance = 2.8
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -35
            while viewer.is_running():
                t0 = time.time()
                control()
                env.step()
                if publisher is not None:
                    publisher.maybe_publish(env)
                viewer.sync()
                dt = env.model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
    finally:
        if publisher is not None:
            publisher.close()
        env.close()


if __name__ == "__main__":
    main()
