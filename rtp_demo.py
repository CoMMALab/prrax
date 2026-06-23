#!/usr/bin/env python3
"""Real-time point-to-point (side-to-side) demo wired into the live DDS stacks.

Pipeline
--------
  perception  ──FilteredPointCloudTopic──▶  this demo  ──JointConfigurationSequenceTopic──▶  controller
  robot       ──ObservedJointConfigurationTopic──▶  this demo (current q / path pacing)

Each leg of the demo plans a straight-line joint-space motion between two
side-to-side configurations, **validates every waypoint of that motion against
the live fused point cloud using pyroffi's RoboGPU collision checker** (robot
collision spheres vs. the cloud, plus self-collision), and only then streams the
planned configurations to the running joint-configuration controller. The
robot's observed joint configuration is used to pace the stream so the arm
actually tracks the planned path before the next leg begins.

Collision backends (``--cc``)
  * ``robogpu`` (default) — OptiX ray-traced sphere/point checker. Requires an
    OptiX-capable driver (``libnvoptix.so``). This is the intended production
    checker.
  * ``cuda``              — pure-CUDA RobotCollisionSpherized SDF checker (no
    OptiX). Functionally equivalent (robot spheres vs. point spheres) and useful
    on platforms without an OptiX driver (e.g. Tegra/Thor).

DDS wire types mirror the robot/controller IDL exactly so discovery matches:
  * subscribe ``JointMsg.ObservedJointConfiguration`` and
    ``PerceptionMsg.FilteredPointCloud`` (mirrors the perception stack);
  * publish ``JointMsg::JointConfigurationSequence`` with Reliable +
    TransientLocal + KeepLast(50) QoS (mirrors joint_configuration_controller.cpp).

Run inside the ``prrax`` conda env. RoboGPU needs CYCLONEDDS_HOME pointing at a
cyclonedds install (the cyclonedds python binding was built against the
pointcloud_perception env):

    CYCLONEDDS_HOME=/home/scoumar/miniconda3/envs/pointcloud_perception \
        python rtp_demo.py --cc cuda          # local test on this machine
    python rtp_demo.py                         # --cc robogpu (default) on OptiX GPU
"""

import argparse
import signal
import sys
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
import jaxlie
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

RESOURCE_ROOT = Path(__file__).resolve().parent / "pyroffi" / "resources" / "panda"
SPHERIZED_URDF = RESOURCE_ROOT / "panda_spherized.urdf"
SRDF = RESOURCE_ROOT / "panda.srdf"


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


class ConfigPublisher:
    """Streams joint configurations to joint_configuration_controller.cpp.

    QoS (Reliable + TransientLocal + KeepLast(50)) matches the controller's
    reader so DDS discovery pairs them up."""

    def __init__(self, participant: DomainParticipant):
        qos = Qos(
            Policy.Reliability.Reliable(duration(seconds=1)),
            Policy.Durability.TransientLocal,
            Policy.History.KeepLast(50),
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
# Collision checking — validate a batch of configs against the live cloud.
# ---------------------------------------------------------------------------
class CollisionValidator:
    """Wraps a pyroffi collision backend; ``free_mask(configs)`` returns a bool
    array (True = collision-free) for an (B, 7) batch validated against the most
    recently set point cloud."""

    def __init__(self, backend: str, r_env: float, max_points: int,
                 self_filter_margin: float = 0.06):
        self.backend = backend
        self.r_env = r_env
        self.max_points = max_points
        self.self_filter_margin = self_filter_margin

        urdf = yourdfpy.URDF.load(str(SPHERIZED_URDF))
        self.robot = pk.Robot.from_urdf(urdf)
        self.coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(SRDF))
        self._world = None  # cached CollGeom for the cuda backend

        # Static geometry for self-filtering the cloud (the perception cloud is
        # NOT body-filtered, so it contains points on the arm itself). Mirrors
        # the vamp demo's filter_self_from_pointcloud step.
        f_local = np.asarray(_spherized_local_geometry(self.coll))  # [K, 4] xyz+r
        NL = self.coll.num_links
        sphere_link = np.arange(f_local.shape[0]) % NL
        self._sf_valid = f_local[:, 3] > 0.0
        # Per valid sphere: owning link name (for yourdfpy FK), local xyz, radius.
        # pyroffi packs sphere k under link (k % num_links); the local xyz is
        # expressed in that link's frame.
        self._sf_urdf = urdf  # reuse the loaded model for CPU forward kinematics
        link_name_per_sphere = [self.coll.link_names[i] for i in sphere_link]
        self._sf_link_names = [
            link_name_per_sphere[i]
            for i in range(len(link_name_per_sphere))
            if self._sf_valid[i]
        ]
        self._sf_unique_links = sorted(set(self._sf_link_names))
        self._sf_link_col = np.array(
            [self._sf_unique_links.index(n) for n in self._sf_link_names], dtype=np.int32
        )
        self._sf_local = f_local[self._sf_valid, :3].astype(np.float64)
        self._sf_radii = f_local[self._sf_valid, 3].astype(np.float32)

        if backend == "robogpu":
            from pyroffi.collision import RoboGPUCollisionChecker

            self.checker = RoboGPUCollisionChecker(self.coll)
            self._far = Sphere.from_center_and_radius(
                center=jnp.array([[1e3, 1e3, 1e3]]), radius=jnp.array([0.01])
            )
        elif backend == "cuda":
            from pyroffi.collision import CUDARobotCollisionChecker

            self.checker = CUDARobotCollisionChecker(self.coll)
        else:
            raise ValueError(f"unknown collision backend: {backend}")

    def _robot_sphere_centers(self, q: np.ndarray) -> np.ndarray:
        """World-frame centers of the robot's collision spheres at config q.

        Uses yourdfpy's CPU forward kinematics (the pyroffi FFI FK is unavailable
        on platforms without the matching kernel build), so self-filtering works
        regardless of the collision backend."""
        self._sf_urdf.update_cfg(np.asarray(q, dtype=np.float64))
        # World transform of each unique owning link.
        T = np.stack(
            [self._sf_urdf.get_transform(name) for name in self._sf_unique_links]
        )  # [U, 4, 4]
        Tp = T[self._sf_link_col]  # [Kv, 4, 4]
        local_h = np.concatenate(
            [self._sf_local, np.ones((len(self._sf_local), 1))], axis=1
        )  # [Kv, 4]
        centers = np.einsum("kij,kj->ki", Tp, local_h)[:, :3]
        return centers.astype(np.float32)

    def self_filter(self, cloud: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Drop cloud points lying inside the robot body at configuration q."""
        if cloud is None or len(cloud) == 0 or q is None:
            return cloud
        centers = self._robot_sphere_centers(q)               # [Kv, 3]
        thresh = (self._sf_radii + self.r_env + self.self_filter_margin) ** 2  # [Kv]
        d2 = ((cloud[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [N, Kv]
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

    def set_cloud(self, cloud: np.ndarray, q_self: np.ndarray = None) -> int:
        if q_self is not None:
            cloud = self.self_filter(np.ascontiguousarray(cloud, np.float32), q_self)
        pts = self._prep_points(cloud)
        pts_j = jnp.asarray(pts)
        if self.backend == "robogpu":
            # RoboGPU keeps self-collision enabled (default) so the verdict
            # covers both point-cloud contact and arm self-collision.
            self.checker.set_world(self._far, point_cloud=pts_j, r_env=self.r_env)
        else:
            self._world = (
                None
                if len(pts) == 0
                else Sphere.from_center_and_radius(
                    center=pts_j,
                    radius=jnp.full((len(pts),), self.r_env, jnp.float32),
                )
            )
        return len(pts)

    def free_mask(self, configs: np.ndarray) -> np.ndarray:
        cfg = jnp.asarray(np.ascontiguousarray(configs, dtype=np.float32))
        if self.backend == "robogpu":
            free = np.asarray(self.checker.check_collision_free(self.robot, cfg))
            return free.reshape(-1).astype(bool)
        # cuda: negative min signed distance over (links, points) ⇒ collision.
        if self._world is None:
            return np.ones(len(configs), dtype=bool)
        d = np.asarray(
            self.checker.compute_world_collision_distance(self.robot, cfg, self._world)
        )
        return d.reshape(len(configs), -1).min(axis=1) > 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
_g_stop = False


def _on_signal(sig, frame):
    global _g_stop
    _g_stop = True


def lerp_path(q0, q1, n):
    """n interpolated configs from q0 to q1 inclusive (n×7)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) * np.asarray(q0)[None, :] + t * np.asarray(q1)[None, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", choices=["robogpu", "cuda"], default="robogpu",
                        help="collision backend (default: robogpu / OptiX)")
    parser.add_argument("--r-env", type=float, default=0.02,
                        help="point-sphere radius for the cloud (m)")
    parser.add_argument("--max-points", type=int, default=20000,
                        help="cap on cloud points fed to the checker")
    parser.add_argument("--self-filter-margin", type=float, default=0.06,
                        help="extra margin (m) when culling on-robot cloud points")
    parser.add_argument("--waypoints", type=int, default=40,
                        help="interpolated configs per leg (validation density)")
    parser.add_argument("--stream-hz", type=float, default=100.0,
                        help="rate the current target is (re)published to the controller")
    parser.add_argument("--reach-tol", type=float, default=0.12,
                        help="per-joint tolerance (rad) for advancing a waypoint")
    parser.add_argument("--goal-tol", type=float, default=0.06,
                        help="per-joint tolerance (rad) for the final waypoint of a leg")
    parser.add_argument("--waypoint-timeout", type=float, default=8.0,
                        help="max seconds to wait for the arm to reach a waypoint")
    parser.add_argument("--cycles", type=int, default=0,
                        help="number of legs to run (0 = forever)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[demo] Loading robot + collision backend '{args.cc}' ...")
    validator = CollisionValidator(args.cc, args.r_env, args.max_points,
                                   args.self_filter_margin)

    print("[demo] Setting up DDS ...")
    participant = DomainParticipant()
    joints_sub = JointConfigSubscriber(participant)
    cloud_sub = CloudSubscriber(participant)
    config_pub = ConfigPublisher(participant)

    # Background streamer: continuously (re)publish the current target so the
    # controller's CommandLink stays "engaged" (it falls back to default after
    # ~2 s of silence) and tracks the latest planned waypoint.
    target_lock = threading.Lock()
    current_target = {"q": None}

    def streamer():
        period = 1.0 / args.stream_hz
        while not _g_stop:
            with target_lock:
                q = current_target["q"]
            if q is not None:
                config_pub.publish(q)
            time.sleep(period)

    def set_target(q):
        with target_lock:
            current_target["q"] = np.asarray(q, dtype=np.float64)

    threading.Thread(target=streamer, daemon=True).start()

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

    # Hold the current pose first so the controller engages without a jump.
    set_target(joints_sub.latest())
    time.sleep(1.0)  # let DDS discovery pair us with the controller's reader

    endpoints = [Q_LEFT, Q_RIGHT]
    leg = 0
    start = joints_sub.latest()

    while not _g_stop and (args.cycles <= 0 or leg < args.cycles):
        goal = endpoints[leg % 2]
        print(f"\n[demo] === Leg {leg}: planning {('LEFT->RIGHT','RIGHT->LEFT')[leg % 2]} ===")

        # 1. Refresh the environment cloud and validate the whole leg.
        cloud, seq = cloud_sub.latest()
        q_obs = joints_sub.latest()
        n_raw = 0 if cloud is None else len(cloud)
        n_pts = validator.set_cloud(cloud, q_self=q_obs)
        path = lerp_path(start, goal, args.waypoints)
        free = validator.free_mask(path)
        n_free = int(free.sum())
        print(f"[demo] Validated {len(path)} waypoints vs cloud seq={seq} "
              f"({n_pts} pts after self-filter from {n_raw}): "
              f"{n_free}/{len(path)} free.")

        if n_free < len(path):
            bad = np.where(~free)[0]
            print(f"[demo] Straight-line path is BLOCKED at waypoints {bad.tolist()}. "
                  f"Holding current pose, skipping this leg.")
            set_target(start)
            time.sleep(1.0)
            leg += 1
            # Retry the same direction next time by not advancing 'start'.
            continue

        # 2. Stream the validated path, paced by the robot's observed motion.
        print(f"[demo] Path clear — streaming {len(path)} configs to controller.")
        for i, wp in enumerate(path):
            if _g_stop:
                break
            tol = args.goal_tol if i == len(path) - 1 else args.reach_tol
            set_target(wp)
            t_wp = time.time()
            while not _g_stop:
                q_obs = joints_sub.latest()
                if q_obs is not None and np.all(np.abs(q_obs - wp) < tol):
                    break
                if time.time() - t_wp > args.waypoint_timeout:
                    print(f"[demo] WARN: waypoint {i} not reached within "
                          f"{args.waypoint_timeout}s (max err "
                          f"{np.max(np.abs(q_obs - wp)):.3f} rad); continuing.")
                    break
                time.sleep(0.01)

        start = joints_sub.latest()
        if start is None:
            start = goal
        leg += 1

    print("\n[demo] Stopping — holding final pose briefly.")
    set_target(joints_sub.latest())
    time.sleep(0.5)
    joints_sub.stop()
    cloud_sub.stop()
    print("[demo] Done.")


if __name__ == "__main__":
    main()
