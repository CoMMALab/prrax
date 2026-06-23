#!/usr/bin/env python3
"""Real-time planning (RTP) demo wired into the live DDS stacks.

Pipeline
--------
  perception  ──FilteredPointCloudTopic──▶  this demo  ──JointConfigurationSequenceTopic──▶  controller
  robot       ──ObservedJointConfigurationTopic──▶  this demo (current q / path pacing)

The arm moves side-to-side **forever**. Every leg is *replanned in real time*:
the demo takes the latest fused point cloud off DDS, self-filters the points
that lie on the arm itself, feeds them to the GPU pRRTC planner as environment
collision spheres, and plans a collision-free joint-space path from the robot's
current configuration to the next side-to-side goal. The planned path is then
streamed (paced by the robot's observed motion) to the joint impedance
controller. If the workspace changes between legs, the next leg is planned
against the new cloud — so an obstacle that moves into the straight-line path is
avoided automatically.

A viser scene (http://localhost:8080, ``--no-viser`` to disable) renders the
live robot configuration, the environment point cloud it plans against, and the
planned end-effector trajectory for the current leg.

Collision / planning backend (``--cc``)
  * ``robogpu`` (default) — host-driven RRT-Connect validated by pyroffi's OptiX
    ray-traced point-cloud checker (``RoboGPUCollisionChecker``). The intended
    production checker; needs pyroffi's robogpu library built. Falls back to
    ``cuda`` if that checker is unavailable.
  * ``cuda``              — the in-kernel CUDA pRRTC planner (``prrtc_plan``). FK
    + collision checking run entirely on the GPU; the live cloud is supplied as
    ``world_spheres``. Works on any CUDA GPU (no OptiX needed).

DDS wire types mirror the robot/controller IDL exactly so discovery matches:
  * subscribe ``JointMsg.ObservedJointConfiguration`` and
    ``PerceptionMsg.FilteredPointCloud`` (mirrors the perception stack);
  * publish ``JointMsg::JointConfigurationSequence`` with BestEffort +
    Volatile + KeepLast(1) QoS (mirrors joint_configuration_controller.cpp).

Run inside the ``prrax`` conda env:

    python rtp_demo.py                         # --cc cuda (default)
    python rtp_demo.py --cc robogpu            # if pyroffi's OptiX checker is built
"""

import argparse
import importlib.util
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- DDS (cyclonedds) -------------------------------------------------------
import cyclonedds.idl as idl
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl.types import array, float32, float64, sequence, uint8, uint32
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from cyclonedds.util import duration

# --- pyroffi (robot model + collision) --------------------------------------
import jax.numpy as jnp
import yourdfpy
import pyroffi as pk
from pyroffi.collision import RobotCollisionSpherized, Sphere
from pyroffi.collision._cuda_collision import _spherized_local_geometry


# ---------------------------------------------------------------------------
# DDS message types — kept byte-compatible with the perception/control stacks.
# ---------------------------------------------------------------------------
# NB: no ``from __future__ import annotations`` — cyclonedds resolves the field
# annotations as live type objects at topic-creation time.

@dataclass
class ObservedJointConfiguration(
    idl.IdlStruct, typename="JointMsg.ObservedJointConfiguration"
):
    """Robot's measured joint configuration: q[7] (rad) + Unix-seconds stamp.
    Mirrors the perception stack's subscriber type exactly."""

    q: array[float64, 7]
    timestamp: float64


@dataclass
class FilteredPointCloud(idl.IdlStruct, typename="PerceptionMsg.FilteredPointCloud"):
    """World-frame fused, filtered cloud published by the perception stack."""

    frame_id: str
    timestamp: float64
    seq: uint32
    num_points: uint32
    points: sequence[float32]
    colors: sequence[uint8]


@dataclass
class Positions7(idl.IdlStruct, typename="JointMsg::Positions7"):
    """One joint configuration: exactly 7 values (rad). Mirrors the C++ IDL
    ``JointMsg::Positions7 { double values[7]; }``."""

    values: array[float64, 7]


@dataclass
class JointConfigurationSequence(
    idl.IdlStruct, typename="JointMsg::JointConfigurationSequence"
):
    """Top-level config message consumed by joint_configuration_controller.cpp;
    the controller reads ``joints[0]`` as the commanded target."""

    joints: sequence[Positions7]


JOINT_TOPIC = "ObservedJointConfigurationTopic"
CLOUD_TOPIC = "FilteredPointCloudTopic"
CONFIG_TOPIC = "JointConfigurationSequenceTopic"
DOF = 7

# Two side-to-side configurations (validated reachable poses for this FR3 cell;
# joint 0 swings ~-0.91 rad to ~+1.35 rad). Mirrors the controller's known pair.
Q_RIGHT = np.array([-0.9106, 0.3887, -0.2971, -1.5381, -0.0355, 1.8566, 0.4519])
Q_LEFT = np.array([1.3526, 0.4781, -0.2965, -1.3773, 0.2660, 1.7507, 0.4519])

# --- Robot base pose -------------------------------------------------------
# The robot base is mounted at a fixed offset from the world/perception frame
# the point cloud is published in. The cloud arrives in world frame, but the
# robot's FK (self-filter + planner collision context) runs in the base frame,
# so incoming clouds are transformed world -> base before use.
#   X: +3.16 cm, Y: -40.00 cm, Z: 0, rotation: 90 deg CCW about Z.
def _make_base_transform(tx, ty, tz, yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = (tx, ty, tz)
    return T


T_WORLD_BASE = _make_base_transform(0.0316, -0.40, 0.0, np.pi / 2.0)
T_BASE_WORLD = np.linalg.inv(T_WORLD_BASE)  # world -> base, for incoming clouds

RESOURCE_ROOT = Path(__file__).resolve().parent / "pyroffi" / "resources" / "panda"
SPHERIZED_URDF = RESOURCE_ROOT / "panda_spherized.urdf"
VISUAL_URDF = RESOURCE_ROOT / "panda.urdf"
SRDF = RESOURCE_ROOT / "panda.srdf"
EE_LINK = "panda_hand"

CUDA_RRTC_DIR = Path(__file__).resolve().parent / "cuda-rrtc" / "jax"

# Default static point-cloud snapshot used by --override-pc-stream (world frame).
DEFAULT_PC_SNAPSHOT = Path(__file__).resolve().parent / "fused_20260623_162357.npz"


_CUDA_RRTC_PKG = "cuda_rrtc_jax"


def _load_cuda_rrtc(submodule: str):
    """Load a cuda-rrtc ``jax`` submodule (e.g. ``"prrtc"``).

    The package dir uses a hyphen so it isn't importable as ``cuda_rrtc``; we
    register a synthetic package rooted at it so the modules' relative imports
    (``from .prrtc import ...``) resolve correctly."""
    import sys
    import types

    if _CUDA_RRTC_PKG not in sys.modules:
        pkg = types.ModuleType(_CUDA_RRTC_PKG)
        pkg.__path__ = [str(CUDA_RRTC_DIR)]
        sys.modules[_CUDA_RRTC_PKG] = pkg

    full = f"{_CUDA_RRTC_PKG}.{submodule}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, CUDA_RRTC_DIR / f"{submodule}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {submodule}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec so relative imports resolve
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# DDS plumbing
# ---------------------------------------------------------------------------
class JointConfigSubscriber:
    """Background reader keeping the robot's most recent joint configuration."""

    def __init__(self, participant: DomainParticipant):
        topic = Topic(participant, JOINT_TOPIC, ObservedJointConfiguration)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self._reader = DataReader(participant, topic, qos=qos)
        self._lock = threading.Lock()
        self._q = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            samples = self._reader.take(N=10)
            if samples:
                q = np.asarray(samples[-1].q, dtype=np.float64)
                with self._lock:
                    self._q = q
            time.sleep(0.005)

    def latest(self):
        with self._lock:
            return None if self._q is None else self._q.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


class CloudSubscriber:
    """Background reader keeping the most recent world-frame point cloud (N×3)."""

    def __init__(self, participant: DomainParticipant):
        topic = Topic(participant, CLOUD_TOPIC, FilteredPointCloud)
        qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=1)),
            Policy.History.KeepLast(1),
        )
        self._reader = DataReader(participant, topic, qos=qos)
        self._lock = threading.Lock()
        self._pts = None
        self._seq = -1
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            samples = self._reader.take(N=4)
            if samples:
                s = samples[-1]
                n = int(s.num_points)
                pts = (
                    np.asarray(s.points, dtype=np.float32).reshape(n, 3)
                    if n > 0
                    else np.empty((0, 3), dtype=np.float32)
                )
                with self._lock:
                    self._pts = pts
                    self._seq = int(s.seq)
            time.sleep(0.01)

    def latest(self):
        with self._lock:
            return (None, -1) if self._pts is None else (self._pts.copy(), self._seq)

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


class StaticCloudSource:
    """Drop-in replacement for ``CloudSubscriber`` that serves a fixed cloud from
    a ``.npz`` snapshot (``points`` [N,3] world-frame) instead of listening on
    DDS. Same interface (``latest`` / ``stop``) so the demo loop is unchanged."""

    def __init__(self, path):
        data = np.load(path)
        self._pts = np.ascontiguousarray(data["points"], dtype=np.float32).reshape(-1, 3)
        print(f"[cloud] Static snapshot '{path}' ({len(self._pts)} pts).")

    def latest(self):
        return self._pts.copy(), 0

    def stop(self):
        pass


class ConfigPublisher:
    """Streams joint configurations to joint_configuration_controller.cpp.

    QoS (BestEffort + Volatile + KeepLast(1)) matches the controller's
    reader so DDS discovery pairs them up."""

    def __init__(self, participant: DomainParticipant):
        qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.Durability.Volatile,
            Policy.History.KeepLast(1),
        )
        topic = Topic(participant, CONFIG_TOPIC, JointConfigurationSequence, qos=qos)
        publisher = Publisher(participant)
        self._writer = DataWriter(publisher, topic, qos=qos)

    def publish(self, q):
        msg = JointConfigurationSequence(
            joints=[Positions7(values=[float(v) for v in q])]
        )
        self._writer.write(msg)


# ---------------------------------------------------------------------------
# Real-time planner — replans each leg against the live point cloud.
# ---------------------------------------------------------------------------
class RealTimePlanner:
    """Plans a collision-free joint path from ``start`` to ``goal`` validated
    against the most recent point cloud.

    The robot's FK + collision-sphere geometry are baked into a static pRRTC
    collision context once; only ``world_spheres`` (the environment cloud) is
    refreshed per leg. The cloud is first self-filtered (points lying on the arm
    are dropped) so the robot doesn't see itself as an obstacle."""

    def __init__(self, backend: str, r_env: float, max_points: int,
                 self_filter_margin: float, step_size: float,
                 max_iterations: int):
        self.backend = backend
        self.r_env = r_env
        self.max_points = max_points
        self.self_filter_margin = self_filter_margin
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.last_filtered_cloud = None  # points actually fed to the checker

        urdf = yourdfpy.URDF.load(str(SPHERIZED_URDF))
        self.robot = pk.Robot.from_urdf(urdf)
        self.coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(SRDF))

        # cuda-rrtc planner + collision-context builder.
        prr = _load_cuda_rrtc("prrtc")
        self._utils = _load_cuda_rrtc("utils")
        self._prrtc_plan = prr.prrtc_plan

        # Static collision context (FK + robot spheres + self-pairs). world_spheres
        # starts empty and is replaced each leg.
        self._ctx = self._utils.build_prrtc_collision_context(
            self.robot, self.coll, []
        )

        # Lightweight analytic checker used to *monitor* the path being executed
        # against the freshest cloud (so a moving obstacle triggers a replan).
        from pyroffi.collision import CUDARobotCollisionChecker
        self._monitor = CUDARobotCollisionChecker(self.coll)

        # robogpu path (host-driven OptiX) — only if pyroffi shipped the checker.
        self._robogpu = None
        if backend == "robogpu":
            try:
                from pyroffi.collision import RoboGPUCollisionChecker  # noqa
                self._robogpu_plan = _load_cuda_rrtc(
                    "prrtc_robogpu"
                ).prrtc_plan_robogpu
                self._robogpu = RoboGPUCollisionChecker(self.coll)
                print("[planner] Using robogpu (OptiX) point-cloud checker.")
            except Exception as e:
                print(f"[planner] robogpu unavailable ({e}); falling back to "
                      f"in-kernel CUDA planner.")
                self.backend = "cuda"

        # --- self-filter geometry (CPU FK on owning links) ------------------
        f_local = np.asarray(_spherized_local_geometry(self.coll))  # [K, 4] xyz+r
        NL = self.coll.num_links
        sphere_link = np.arange(f_local.shape[0]) % NL
        self._sf_valid = f_local[:, 3] > 0.0
        self._sf_urdf = urdf
        link_name_per_sphere = [self.coll.link_names[i] for i in sphere_link]
        self._sf_link_names = [
            link_name_per_sphere[i]
            for i in range(len(link_name_per_sphere))
            if self._sf_valid[i]
        ]
        self._sf_unique_links = sorted(set(self._sf_link_names))
        self._sf_link_col = np.array(
            [self._sf_unique_links.index(n) for n in self._sf_link_names],
            dtype=np.int32,
        )
        self._sf_local = f_local[self._sf_valid, :3].astype(np.float64)
        self._sf_radii = f_local[self._sf_valid, 3].astype(np.float32)

    # -- frame conversion ----------------------------------------------------
    @staticmethod
    def _world_to_base(cloud: np.ndarray) -> np.ndarray:
        """Transform a world-frame cloud into the robot base frame (FK frame)."""
        if cloud is None or len(cloud) == 0:
            return cloud
        cloud = np.ascontiguousarray(cloud, dtype=np.float32)
        R = T_BASE_WORLD[:3, :3]
        t = T_BASE_WORLD[:3, 3]
        return (cloud @ R.T.astype(np.float32)) + t.astype(np.float32)

    # -- self-filtering ------------------------------------------------------
    def _robot_sphere_centers(self, q: np.ndarray) -> np.ndarray:
        """World-frame centers of the robot's collision spheres at config q
        (yourdfpy CPU forward kinematics)."""
        self._sf_urdf.update_cfg(np.asarray(q, dtype=np.float64))
        T = np.stack(
            [self._sf_urdf.get_transform(name) for name in self._sf_unique_links]
        )  # [U, 4, 4]
        Tp = T[self._sf_link_col]  # [Kv, 4, 4]
        local_h = np.concatenate(
            [self._sf_local, np.ones((len(self._sf_local), 1))], axis=1
        )
        centers = np.einsum("kij,kj->ki", Tp, local_h)[:, :3]
        return centers.astype(np.float32)

    def self_filter(self, cloud: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Drop cloud points lying inside the robot body at configuration q."""
        if cloud is None or len(cloud) == 0 or q is None:
            return cloud
        centers = self._robot_sphere_centers(q)
        thresh = (self._sf_radii + self.r_env + self.self_filter_margin) ** 2
        d2 = ((cloud[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        keep = np.all(d2 > thresh[None, :], axis=1)
        return cloud[keep]

    def _prep_points(self, cloud: np.ndarray) -> np.ndarray:
        if cloud is None or len(cloud) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if len(cloud) > self.max_points:
            idx = np.random.default_rng(0).choice(
                len(cloud), self.max_points, replace=False
            )
            cloud = cloud[idx]
        return np.ascontiguousarray(cloud, dtype=np.float32)

    # -- planning ------------------------------------------------------------
    def replan(self, start, goal, cloud, q_self):
        """Plan a path ``start → goal`` around the (self-filtered) cloud.

        Returns ``(path[N,7] | None, n_pts, n_raw)``. ``path`` is ``None`` when
        no collision-free path was found."""
        n_raw = 0 if cloud is None else len(cloud)
        cloud = self._world_to_base(cloud)
        if cloud is not None and q_self is not None:
            cloud = self.self_filter(
                np.ascontiguousarray(cloud, np.float32), q_self
            )
        pts = self._prep_points(cloud)
        n_pts = len(pts)
        self.last_filtered_cloud = pts

        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        if self.backend == "robogpu" and self._robogpu is not None:
            res = self._robogpu_plan(
                start_config=jnp.asarray(start),
                goal_configs=jnp.asarray(goal).reshape(1, -1),
                robot=self.robot,
                robogpu_checker=self._robogpu,
                point_cloud=jnp.asarray(pts) if n_pts else None,
                r_env=self.r_env,
                step_size=self.step_size,
                max_iterations=self.max_iterations,
            )
        else:
            world_spheres = (
                np.concatenate(
                    [pts, np.full((n_pts, 1), self.r_env, np.float32)], axis=1
                )
                if n_pts
                else np.zeros((0, 4), dtype=np.float32)
            )
            self._ctx["world_spheres"] = world_spheres
            res = self._prrtc_plan(
                start_config=jnp.asarray(start),
                goal_configs=jnp.asarray(goal).reshape(1, -1),
                collision_context=self._ctx,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
            )

        if not bool(res.solved) or res.path is None:
            return None, n_pts, n_raw
        path = np.asarray(res.path, dtype=np.float64)
        return path, n_pts, n_raw

    # -- monitoring ----------------------------------------------------------
    def clearance(self, configs, cloud, q_self):
        """Min robot-sphere clearance (m) per config against the self-filtered
        ``cloud`` (point spheres of radius ``r_env``). ``+inf`` if the cloud is
        empty. Used to detect when a moving obstacle has invalidated the path
        currently being executed. Updates ``last_filtered_cloud`` so the viewer
        can show the live cloud being checked."""
        configs = np.atleast_2d(np.asarray(configs, dtype=np.float32))
        cloud = self._world_to_base(cloud)
        if cloud is not None and q_self is not None:
            cloud = self.self_filter(np.ascontiguousarray(cloud, np.float32), q_self)
        pts = self._prep_points(cloud)
        self.last_filtered_cloud = pts
        if len(pts) == 0:
            return np.full(len(configs), np.inf, dtype=np.float32)
        world = Sphere.from_center_and_radius(
            center=jnp.asarray(pts),
            radius=jnp.full((len(pts),), self.r_env, jnp.float32),
        )
        d = np.asarray(
            self._monitor.compute_world_collision_distance(
                self.robot, jnp.asarray(configs), world
            )
        )
        return d.reshape(len(configs), -1).min(axis=1)


def densify_path(path: np.ndarray, max_step: float = 0.08) -> np.ndarray:
    """Subdivide a sparse waypoint path so consecutive configs differ by at most
    ``max_step`` rad/joint — gives the impedance controller a smooth target
    stream and lets the pacing loop track progress finely."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Viser visualization — live robot + point cloud + planned EE trajectory.
# ---------------------------------------------------------------------------
class Viz:
    """Renders the demo in a viser scene: the live robot configuration, the
    environment point cloud it is planning against, and the planned end-effector
    trajectory for the current leg."""

    def __init__(self, port: int = 8080):
        import viser
        from viser.extras import ViserUrdf

        self._urdf = yourdfpy.URDF.load(str(VISUAL_URDF))  # FK for the EE path
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=3.0, height=3.0)
        self._robot = ViserUrdf(
            self.server, self._urdf, root_node_name="/robot"
        )
        self._robot.update_cfg(np.asarray(Q_LEFT))
        self._cloud_handle = None
        self._path_handle = None
        self._endpoints_handle = None
        print(f"[viz] viser server at http://localhost:{port}")

    def set_config(self, q):
        if q is not None:
            self._robot.update_cfg(np.asarray(q, dtype=np.float64))

    def set_cloud(self, pts):
        if pts is None or len(pts) == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        colors = np.tile(np.array([80, 160, 255], np.uint8), (len(pts), 1))
        self._cloud_handle = self.server.scene.add_point_cloud(
            "/world/cloud", points=np.asarray(pts, np.float32),
            colors=colors, point_size=0.006,
        )

    def _ee_positions(self, path):
        pos = np.empty((len(path), 3), dtype=np.float32)
        for i, q in enumerate(path):
            self._urdf.update_cfg(np.asarray(q, dtype=np.float64))
            pos[i] = self._urdf.get_transform(EE_LINK)[:3, 3]
        return pos

    def set_path(self, path):
        """Draw the planned EE trajectory (workspace polyline) for the leg."""
        if path is None or len(path) < 2:
            return
        pos = self._ee_positions(path)
        self._path_handle = self.server.scene.add_spline_catmull_rom(
            "/world/plan", positions=pos, color=(255, 90, 40), line_width=4.0,
        )
        self._endpoints_handle = self.server.scene.add_point_cloud(
            "/world/plan_pts", points=pos,
            colors=np.tile(np.array([255, 220, 0], np.uint8), (len(pos), 1)),
            point_size=0.012,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
_g_stop = False


def _on_signal(sig, frame):
    global _g_stop
    _g_stop = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", choices=["cuda", "robogpu"], default="robogpu",
                        help="planning/collision backend (default: robogpu / OptiX)")
    parser.add_argument("--r-env", type=float, default=0.04,
                        help="point-sphere radius for the cloud (m); also the "
                             "planned standoff from obstacle points")
    parser.add_argument("--max-points", type=int, default=20000,
                        help="cap on cloud points fed to the planner")
    parser.add_argument("--self-filter-margin", type=float, default=0.06,
                        help="extra margin (m) when culling on-robot cloud points")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="pRRTC max extension step in config space (rad)")
    parser.add_argument("--max-iterations", type=int, default=20000,
                        help="pRRTC iteration cap per replan")
    parser.add_argument("--max-step", type=float, default=0.08,
                        help="max per-joint delta (rad) between streamed configs")
    parser.add_argument("--stream-hz", type=float, default=100.0,
                        help="rate the current target is (re)published to the controller")
    parser.add_argument("--reach-tol", type=float, default=0.12,
                        help="per-joint tolerance (rad) for advancing a waypoint")
    parser.add_argument("--goal-tol", type=float, default=0.06,
                        help="per-joint tolerance (rad) for the final waypoint of a leg")
    parser.add_argument("--waypoint-timeout", type=float, default=8.0,
                        help="max seconds to wait for the arm to reach a waypoint")
    parser.add_argument("--monitor-hz", type=float, default=10.0,
                        help="rate to re-check the executing path vs the live cloud")
    parser.add_argument("--safety-margin", type=float, default=0.0,
                        help="replan if the remaining path's clearance to the "
                             "live cloud drops below this (m)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="number of legs to run (0 = forever)")
    parser.add_argument("--override-pc-stream", nargs="?", type=str,
                        const=str(DEFAULT_PC_SNAPSHOT), default=None,
                        metavar="NPZ",
                        help="ignore the DDS point-cloud stream and plan against a "
                             "static .npz snapshot (points[N,3] world frame); "
                             f"defaults to {DEFAULT_PC_SNAPSHOT.name} if no path given")
    parser.add_argument("--no-viser", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument("--viser-port", type=int, default=8080,
                        help="port for the viser server")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[demo] Loading robot + planner backend '{args.cc}' ...")
    planner = RealTimePlanner(
        args.cc, args.r_env, args.max_points, args.self_filter_margin,
        args.step_size, args.max_iterations,
    )

    viz = None
    if not args.no_viser:
        print("[demo] Starting viser visualization ...")
        viz = Viz(args.viser_port)

    print("[demo] Setting up DDS ...")
    participant = DomainParticipant()
    joints_sub = JointConfigSubscriber(participant)
    if args.override_pc_stream is not None:
        cloud_sub = StaticCloudSource(args.override_pc_stream)
    else:
        cloud_sub = CloudSubscriber(participant)
    config_pub = ConfigPublisher(participant)

    # Background streamer: continuously (re)publish the current target so the
    # controller's CommandLink stays "engaged" (it falls back to default after
    # ~2 s of silence) and tracks the latest planned waypoint.
    target_lock = threading.Lock()
    current_target = {"q": None}

    def streamer():
        # Tight, steady publish loop: nothing but the DDS write happens here so
        # the controller sees a regular command cadence. Irregular timing (e.g.
        # viser rendering or a GPU stall sharing this thread) makes the command
        # stream look like velocity spikes and trips the Franka reflex
        # `communication_constraints_violation`. We pace off an absolute monotonic
        # schedule so per-tick work cost can't accumulate drift.
        period = 1.0 / args.stream_hz
        next_t = time.monotonic()
        while not _g_stop:
            with target_lock:
                q = current_target["q"]
            if q is not None:
                config_pub.publish(q)
            next_t += period
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()  # we fell behind; resync, don't burst

    def render_loop():
        # viser rendering at a modest, fixed rate — kept off the publish path.
        if viz is None:
            return
        while not _g_stop:
            viz.set_config(joints_sub.latest())
            time.sleep(1.0 / 30.0)

    def set_target(q):
        with target_lock:
            current_target["q"] = np.asarray(q, dtype=np.float64)

    threading.Thread(target=streamer, daemon=True).start()
    threading.Thread(target=render_loop, daemon=True).start()

    # Wait for the first joint configuration + cloud so we plan from the real state.
    print("[demo] Waiting for observed joint configuration and point cloud ...")
    t0 = time.time()
    while not _g_stop:
        q_now = joints_sub.latest()
        cloud, seq = cloud_sub.latest()
        if q_now is not None and cloud is not None:
            print(f"[demo] Got q0 and cloud (seq={seq}, {len(cloud)} pts).")
            break
        if time.time() - t0 > 15.0:
            print("[demo] ERROR: no joint config / cloud within 15 s. Are the "
                  "perception + controller topics streaming? (probe with "
                  "`cyclonedds ls`)")
            return
        time.sleep(0.1)
    if _g_stop:
        return

    # Warm up the planner before we start commanding: the first pRRTC/clearance
    # call triggers JAX/GPU JIT compilation and can block for seconds. Doing it
    # now — while no path is being streamed — keeps that stall out of the live
    # command loop, where it would starve the publish cadence and risk a Franka
    # `communication_constraints_violation`.
    q_warm = joints_sub.latest()
    cloud_warm, _ = cloud_sub.latest()
    print("[demo] Warming up planner (JIT compile) ...")
    t_warm = time.time()
    planner.replan(q_warm, q_warm, cloud_warm, q_warm)
    planner.clearance(np.atleast_2d(q_warm), cloud_warm, q_warm)
    print(f"[demo] Planner warm ({(time.time() - t_warm)*1e3:.0f} ms).")

    # Hold the current pose first so the controller engages without a jump.
    set_target(joints_sub.latest())
    time.sleep(1.0)  # let DDS discovery pair us with the controller's reader

    def execute(path):
        """Stream a densified path, paced by observed motion while continuously
        monitoring the *live* cloud. Returns 'reached', 'blocked' (a moving
        obstacle invalidated the remaining path → caller should replan), or
        'stop'.

        The collision monitor runs on its own thread so the blocking JAX/GPU
        ``clearance()`` call never stalls the waypoint-advance loop. Keeping that
        loop a tight ~100 Hz check means the streamed impedance target is released
        the instant the arm reaches a waypoint, instead of lagging behind the
        per-tick clearance compute (which is what made motion slow and choppy)."""
        mon_period = 1.0 / args.monitor_hz
        blocked = threading.Event()
        done = threading.Event()
        idx_lock = threading.Lock()
        cur_idx = {"i": 0}

        def monitor():
            # Re-check the as-yet-unreached remainder against the live cloud on a
            # steady cadence; flag 'blocked' if clearance drops below the margin.
            next_t = time.monotonic()
            while not done.is_set() and not _g_stop:
                with idx_lock:
                    i = cur_idx["i"]
                q_obs = joints_sub.latest()
                cloud_live, _ = cloud_sub.latest()
                clr = planner.clearance(path[i:], cloud_live, q_obs)
                if viz is not None:
                    viz.set_cloud(planner.last_filtered_cloud)
                if float(np.min(clr)) < args.safety_margin:
                    blocked.set()
                    return
                next_t += mon_period
                dt = next_t - time.monotonic()
                if dt > 0:
                    done.wait(dt)
                else:
                    next_t = time.monotonic()  # fell behind; resync, don't burst

        mon_thread = threading.Thread(target=monitor, daemon=True)
        mon_thread.start()
        try:
            for i, wp in enumerate(path):
                if _g_stop:
                    return "stop"
                if blocked.is_set():
                    return "blocked"
                with idx_lock:
                    cur_idx["i"] = i
                tol = args.goal_tol if i == len(path) - 1 else args.reach_tol
                set_target(wp)
                t_wp = time.time()
                while not _g_stop:
                    if blocked.is_set():
                        return "blocked"
                    q_obs = joints_sub.latest()
                    if q_obs is not None and np.all(np.abs(q_obs - wp) < tol):
                        break
                    if time.time() - t_wp > args.waypoint_timeout:
                        print(f"[demo] WARN: waypoint {i} not reached within "
                              f"{args.waypoint_timeout}s (max err "
                              f"{np.max(np.abs(q_obs - wp)):.3f} rad); continuing.")
                        break
                    time.sleep(0.01)
            return "reached"
        finally:
            done.set()
            mon_thread.join(timeout=1.0)

    endpoints = [Q_LEFT, Q_RIGHT]
    names = ("LEFT->RIGHT", "RIGHT->LEFT")
    leg = 0

    while not _g_stop and (args.cycles <= 0 or leg < args.cycles):
        goal = endpoints[leg % 2]
        print(f"\n[demo] === Leg {leg}: {names[leg % 2]} ===")

        # Replan from the current state until the goal is reached. A leg may be
        # replanned several times if a moving obstacle blocks the path mid-motion.
        reached = False
        while not _g_stop and not reached:
            cloud, seq = cloud_sub.latest()
            q_obs = joints_sub.latest()
            start = q_obs if q_obs is not None else goal
            t_plan = time.time()
            path, n_pts, n_raw = planner.replan(start, goal, cloud, q_obs)
            dt = time.time() - t_plan
            if viz is not None:
                viz.set_cloud(planner.last_filtered_cloud)

            if path is None:
                print(f"[demo] No collision-free path vs cloud seq={seq} "
                      f"({n_pts}/{n_raw} pts) in {dt*1e3:.0f} ms. Holding, retrying.")
                set_target(start)
                time.sleep(0.3)
                continue

            path = densify_path(path, args.max_step)
            if viz is not None:
                viz.set_path(path)
            print(f"[demo] Planned {len(path)} configs vs cloud seq={seq} "
                  f"({n_pts}/{n_raw} pts) in {dt*1e3:.0f} ms — streaming.")

            status = execute(path)
            if status == "stop":
                break
            if status == "blocked":
                print("[demo] Obstacle moved into the path — replanning from "
                      "current pose.")
                continue
            reached = True

        leg += 1

    print("\n[demo] Stopping — holding final pose briefly.")
    set_target(joints_sub.latest())
    time.sleep(0.5)
    joints_sub.stop()
    cloud_sub.stop()
    print("[demo] Done.")


if __name__ == "__main__":
    main()
