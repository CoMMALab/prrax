#!/usr/bin/env python3
"""Mock DDS harness for testing rtp_demo.py without the real robot/perception.

Publishes the two topics rtp_demo.py consumes and consumes the one it produces:

  * publishes ``PerceptionMsg.FilteredPointCloud`` on FilteredPointCloudTopic —
    a static "obstacle" point cloud (a vertical wall slab) plus optional motion
    so the demo replans against a changing scene;
  * subscribes ``JointMsg::JointConfigurationSequence`` on
    JointConfigurationSequenceTopic, reads ``joints[0]`` as the commanded target,
    and simulates a first-order joint impedance controller tracking it;
  * publishes the simulated ``JointMsg.ObservedJointConfiguration`` on
    ObservedJointConfigurationTopic.

Run alongside rtp_demo.py (same DDS domain) in the prrax env:

    python rtp_sim.py            # then, in another shell:  python rtp_demo.py
"""

import argparse
import signal
import threading
import time
from dataclasses import dataclass

import numpy as np

import cyclonedds.idl as idl
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl.types import array, float32, float64, sequence, uint8, uint32
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from cyclonedds.util import duration

# Reuse the exact wire types from the demo so discovery/typing match.
from rtp_demo import (
    CLOUD_TOPIC,
    CONFIG_TOPIC,
    JOINT_TOPIC,
    Q_LEFT,
    Q_RIGHT,
    FilteredPointCloud,
    JointConfigurationSequence,
    ObservedJointConfiguration,
)

_g_stop = False


def _on_signal(sig, frame):
    global _g_stop
    _g_stop = True


def make_wall(n, x, y_center, half_w, z0, z1, jitter=0.005, rng=None):
    """A slab of points approximating a vertical wall in the y-z plane at depth x."""
    rng = rng or np.random.default_rng(0)
    ys = rng.uniform(y_center - half_w, y_center + half_w, n)
    zs = rng.uniform(z0, z1, n)
    xs = np.full(n, x) + rng.uniform(-jitter, jitter, n)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=3000, help="obstacle cloud size")
    ap.add_argument("--cloud-hz", type=float, default=10.0)
    ap.add_argument("--obs-hz", type=float, default=200.0, help="observed-state publish rate")
    ap.add_argument("--gain", type=float, default=6.0,
                    help="impedance tracking gain (1/s); higher = stiffer")
    ap.add_argument("--obstacle", choices=["none", "wall", "moving"], default="wall",
                    help="obstacle scene to publish")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    participant = DomainParticipant()

    # --- observed joint state (sim) ----------------------------------------
    obs_topic = Topic(participant, JOINT_TOPIC, ObservedJointConfiguration)
    obs_qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
    obs_writer = DataWriter(Publisher(participant), obs_topic, qos=obs_qos)

    # --- command input (controller target) ---------------------------------
    cmd_qos = Qos(
        Policy.Reliability.Reliable(duration(seconds=1)),
        Policy.Durability.TransientLocal,
        Policy.History.KeepLast(50),
    )
    cmd_topic = Topic(participant, CONFIG_TOPIC, JointConfigurationSequence, qos=cmd_qos)
    cmd_reader = DataReader(Subscriber(participant), cmd_topic, qos=cmd_qos)

    # --- cloud output -------------------------------------------------------
    cloud_qos = Qos(
        Policy.Reliability.Reliable(duration(seconds=1)),
        Policy.History.KeepLast(1),
    )
    cloud_topic = Topic(participant, CLOUD_TOPIC, FilteredPointCloud, qos=cloud_qos)
    cloud_writer = DataWriter(Publisher(participant), cloud_topic, qos=cloud_qos)

    state = {"q": np.array(Q_LEFT, dtype=np.float64), "target": np.array(Q_LEFT, dtype=np.float64)}
    lock = threading.Lock()

    # Command reader thread: latest joints[0] becomes the impedance target.
    def cmd_loop():
        while not _g_stop:
            samples = cmd_reader.take(N=20)
            if samples:
                s = samples[-1]
                if s.joints:
                    tgt = np.asarray(s.joints[0].values, dtype=np.float64)
                    with lock:
                        state["target"] = tgt
            time.sleep(0.002)

    threading.Thread(target=cmd_loop, daemon=True).start()

    # Cloud publisher thread.
    def cloud_loop():
        rng = np.random.default_rng(0)
        seq = 0
        period = 1.0 / args.cloud_hz
        t_start = time.time()
        while not _g_stop:
            if args.obstacle == "none":
                pts = np.empty((0, 3), dtype=np.float32)
            else:
                y = 0.0
                if args.obstacle == "moving":
                    # slab drifts side-to-side across the swept region.
                    y = 0.35 * np.sin(0.2 * (time.time() - t_start))
                pts = make_wall(args.points, x=0.55, y_center=y, half_w=0.12,
                                z0=0.25, z1=0.75, rng=rng)
            n = len(pts)
            msg = FilteredPointCloud(
                frame_id="world",
                timestamp=time.time(),
                seq=seq,
                num_points=n,
                points=pts.reshape(-1).tolist(),
                colors=[],
            )
            cloud_writer.write(msg)
            seq += 1
            time.sleep(period)

    threading.Thread(target=cloud_loop, daemon=True).start()

    # Main loop: integrate the first-order impedance sim + publish observed q.
    print(f"[sim] Publishing obstacle='{args.obstacle}' cloud and simulating "
          f"impedance controller (gain={args.gain}).")
    period = 1.0 / args.obs_hz
    last = time.time()
    while not _g_stop:
        now = time.time()
        dt = now - last
        last = now
        with lock:
            tgt = state["target"]
            q = state["q"]
            # first-order tracking: q += gain*(target-q)*dt, clamped to dt<=1.
            q = q + (tgt - q) * min(1.0, args.gain * dt)
            state["q"] = q
        obs_writer.write(ObservedJointConfiguration(q=q.tolist(), timestamp=now))
        time.sleep(period)

    print("\n[sim] Done.")


if __name__ == "__main__":
    main()
