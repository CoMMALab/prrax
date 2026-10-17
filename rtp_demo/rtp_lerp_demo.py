#!/usr/bin/env python3
"""Straight-line (LERP) side-to-side demo — the no-planner baseline for rtp_demo.

Same DDS wiring as ``rtp_demo.py`` (subscribes the robot's observed joint state,
publishes ``JointMsg::JointConfigurationSequence`` to the controller), but the
"plan" for each leg is just a **linear interpolation in joint space** from the
robot's current configuration to the next side-to-side goal. There is no point
cloud, no self-filtering, and no collision checking: the arm swings straight
through whatever is in the way. Useful as a baseline to compare against the
collision-free pRRTC planner.

Run inside the ``prrax`` conda env (alongside ``rtp_sim.py`` to simulate the
robot, or against the real controller):

    python rtp_lerp_demo.py
"""

import argparse
import signal
import threading
import time

import numpy as np

from cyclonedds.domain import DomainParticipant

# Reuse the demo's DDS plumbing + wire types so discovery/typing match exactly.
from rtp_demo import (
    Q_LEFT,
    Q_RIGHT,
    ConfigPublisher,
    JointConfigSubscriber,
    densify_path,
)

_g_stop = False


def _on_signal(sig, frame):
    global _g_stop
    _g_stop = True


def lerp_path(start, goal, max_step):
    """Straight line in joint space from ``start`` to ``goal``, subdivided so
    consecutive configs differ by at most ``max_step`` rad/joint."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return densify_path(np.stack([start, goal]), max_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--cycles", type=int, default=0,
                        help="number of legs to run (0 = forever)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print("[demo] Setting up DDS ...")
    participant = DomainParticipant()
    joints_sub = JointConfigSubscriber(participant)
    config_pub = ConfigPublisher(participant)

    # Background streamer: continuously (re)publish the current target so the
    # controller's CommandLink stays "engaged" (it falls back to default after
    # ~2 s of silence) and tracks the latest waypoint.
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

    # Wait for the first joint configuration so we LERP from the real state.
    print("[demo] Waiting for observed joint configuration ...")
    t0 = time.time()
    while not _g_stop:
        if joints_sub.latest() is not None:
            print("[demo] Got q0.")
            break
        if time.time() - t0 > 15.0:
            print("[demo] ERROR: no joint config within 15 s. Is the controller "
                  "topic streaming? (probe with `cyclonedds ls`)")
            return
        time.sleep(0.1)
    if _g_stop:
        return

    # Hold the current pose first so the controller engages without a jump.
    set_target(joints_sub.latest())
    time.sleep(1.0)  # let DDS discovery pair us with the controller's reader

    def execute(path):
        """Stream a densified path, paced by observed motion. Returns 'reached'
        or 'stop'."""
        for i, wp in enumerate(path):
            if _g_stop:
                return "stop"
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
        return "reached"

    endpoints = [Q_LEFT, Q_RIGHT]
    names = ("LEFT->RIGHT", "RIGHT->LEFT")
    leg = 0

    while not _g_stop and (args.cycles <= 0 or leg < args.cycles):
        goal = endpoints[leg % 2]
        print(f"\n[demo] === Leg {leg}: {names[leg % 2]} (LERP) ===")

        q_obs = joints_sub.latest()
        start = q_obs if q_obs is not None else endpoints[(leg + 1) % 2]
        path = lerp_path(start, goal, args.max_step)
        print(f"[demo] LERP {len(path)} configs from current pose — streaming.")

        if execute(path) == "stop":
            break
        leg += 1

    print("\n[demo] Stopping — holding final pose briefly.")
    set_target(joints_sub.latest())
    time.sleep(0.5)
    joints_sub.stop()
    print("[demo] Done.")


if __name__ == "__main__":
    main()
