#!/usr/bin/env python3
"""Run an OpenPI policy server against an SO-101 through LeRobot's hardware API."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower import SO101FollowerConfig
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
# Conservative deployment envelope from the SO101-60 action q01/q99 used to train this checkpoint.
ACTION_Q01 = np.array([-23.05217, -99.54145, -51.98219, 33.97929, -42.88221, 0.63827], dtype=np.float32)
ACTION_Q99 = np.array([36.92817, 49.96341, 99.96682, 84.16307, 13.03409, 29.86958], dtype=np.float32)


def camera_source(value: str) -> int | Path:
    return int(value) if value.isdigit() else Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower").expanduser(),
    )
    parser.add_argument("--wrist-camera", default="0")
    parser.add_argument("--front-camera", default="2")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--prompt", default="Grab the blue pen and place it into the black box")
    parser.add_argument("--execute", action="store_true", help="Actually send predicted targets to the motors.")
    parser.add_argument("--confirm-motion", default="NO", help="Must be YES together with --execute.")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Disable the default confirm-every-action step mode. Intended only after step-mode validation.",
    )
    parser.add_argument("--confirm-continuous", default="NO", help="Must be YES together with --continuous.")
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Prefetch the next action chunk on a background WebSocket connection.",
    )
    parser.add_argument(
        "--async-prefetch-threshold",
        type=int,
        default=None,
        help="Start background inference with this many actions left (default: min(40, horizon)).",
    )
    parser.add_argument(
        "--async-transition-steps",
        type=int,
        default=3,
        help="Blend this many targets at each asynchronously prefetched chunk boundary.",
    )
    parser.add_argument(
        "--async-result-timeout",
        type=float,
        default=10.0,
        help="Abort after this many seconds if an async result is unavailable at a chunk boundary.",
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Use JAX denoising-level Real-Time Chunking with measured-delay queue replacement.",
    )
    parser.add_argument(
        "--rtc-queue-threshold",
        type=int,
        default=30,
        help="Start an RTC request when this many executable actions remain.",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=20,
        help="End of the old-chunk prefix guidance window.",
    )
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--no-clip-training-range", action="store_true")
    return parser.parse_args()


def make_request(observation: dict, prompt: str) -> dict:
    state = np.asarray([observation[name] for name in ACTION_NAMES], dtype=np.float32)
    return {
        "observation.images.front": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.asarray(observation["front"]), 224, 224)
        ),
        "observation.images.wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.asarray(observation["wrist"]), 224, 224)
        ),
        "observation.state": state,
        "prompt": prompt,
    }


def make_rtc_request(
    observation: dict,
    prompt: str,
    prev_actions: np.ndarray,
    *,
    inference_delay: int,
    execution_horizon: int,
    max_guidance_weight: float,
) -> dict:
    request = make_request(observation, prompt)
    request["rtc"] = {
        "prev_actions": np.asarray(prev_actions, dtype=np.float32),
        "inference_delay": inference_delay,
        "execution_horizon": execution_horizon,
        "max_guidance_weight": max_guidance_weight,
    }
    return request


def validate_chunk(result: dict) -> np.ndarray:
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (50, 6):
        raise ValueError(f"Expected actions shape (50, 6), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned NaN or Inf actions")
    return actions


def current_state(observation: dict) -> np.ndarray:
    return np.asarray([observation[name] for name in ACTION_NAMES], dtype=np.float32)


def limit_target(target: np.ndarray, state: np.ndarray, max_delta: float) -> np.ndarray:
    return np.clip(target, state - max_delta, state + max_delta)


def print_step_preview(
    state: np.ndarray,
    raw_target: np.ndarray,
    range_target: np.ndarray,
    safe_target: np.ndarray,
    *,
    chunk_index: int,
) -> None:
    print(f"\nAction chunk step {chunk_index}")
    print(f"{'joint':<20} {'current':>10} {'policy':>10} {'qclip':>10} {'send':>10} {'delta':>10}")
    for name, current, raw, clipped, safe in zip(
        ACTION_NAMES, state, raw_target, range_target, safe_target, strict=True
    ):
        print(
            f"{name:<20} {current:>10.3f} {raw:>10.3f} {clipped:>10.3f} "
            f"{safe:>10.3f} {safe - current:>+10.3f}"
        )


def infer_actions(client, observation: dict, args: argparse.Namespace) -> np.ndarray:
    return process_inference_result(client.infer(make_request(observation, args.prompt)), args)


def process_inference_result(result: dict, args: argparse.Namespace) -> np.ndarray:
    actions = validate_chunk(result)
    if not args.no_clip_training_range:
        actions = np.clip(actions, ACTION_Q01, ACTION_Q99)
    return actions


@dataclass(frozen=True)
class AsyncInferenceResult:
    result: dict | None
    latency_s: float
    error: BaseException | None = None


class AsyncPolicyWorker:
    """Own a WebSocket client and perform at most one inference request at a time."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._requests: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[AsyncInferenceResult] = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._run, name="openpi-async-inference", daemon=True)

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def start(self) -> None:
        self._thread.start()

    def submit(self, request: dict) -> bool:
        if self.busy:
            return False
        self._busy.set()
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def poll(self) -> AsyncInferenceResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self._requests.put_nowait(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        client = websocket_client_policy.WebsocketClientPolicy(self._host, self._port)
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.perf_counter()
            try:
                result = AsyncInferenceResult(client.infer(request), time.perf_counter() - started)
            except BaseException as error:  # Surface background failures on the control thread.
                result = AsyncInferenceResult(None, time.perf_counter() - started, error)
            self._results.put(result)
            self._busy.clear()


def blend_chunk(actions: np.ndarray, previous_target: np.ndarray, transition_steps: int) -> np.ndarray:
    if transition_steps <= 0:
        return actions
    blended = actions.copy()
    count = min(transition_steps, len(blended))
    for index in range(count):
        weight = (index + 1) / (count + 1)
        blended[index] = (1.0 - weight) * previous_target + weight * blended[index]
    return blended


def run_step_mode(robot, client, args: argparse.Namespace) -> None:
    sent = 0
    raw_actions = None
    action_index = 0
    while sent < args.steps:
        observation = robot.get_observation()
        state = current_state(observation)
        if raw_actions is None or action_index >= args.open_loop_horizon:
            raw_actions = validate_chunk(client.infer(make_request(observation, args.prompt)))
            action_index = 0
            print(f"Received a new action chunk; reviewing up to {args.open_loop_horizon} actions.")
        raw_target = raw_actions[action_index]
        range_target = raw_target if args.no_clip_training_range else np.clip(raw_target, ACTION_Q01, ACTION_Q99)
        safe_target = limit_target(range_target, state, args.max_relative_target)
        print_step_preview(state, raw_target, range_target, safe_target, chunk_index=action_index)
        answer = input("[s] send and next  [r] replan  [q] quit: ").strip().lower()
        if answer.lower() == "q":
            print("Stopped without sending the previewed target.")
            return
        if answer == "r":
            raw_actions = None
            print("Remaining chunk discarded; replanning from a fresh observation.")
            continue
        if answer != "s":
            print("Not sent. Choose s, r, or q.")
            continue
        actually_sent = robot.send_action(dict(zip(ACTION_NAMES, safe_target, strict=True)))
        sent += 1
        action_index += 1
        print(f"Sent action {sent}/{args.steps}:", actually_sent)


def run_continuous(robot, client, args: argparse.Namespace, actions: np.ndarray) -> None:
    period = 1.0 / args.control_fps
    action_index = 0
    for step in range(args.steps):
        loop_start = time.perf_counter()
        if step > 0 and action_index >= args.open_loop_horizon:
            observation = robot.get_observation()
            actions = infer_actions(client, observation, args)
            action_index = 0
        target = actions[action_index]
        robot.send_action(dict(zip(ACTION_NAMES, target, strict=True)))
        action_index += 1
        time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))


def run_async_continuous(robot, args: argparse.Namespace, actions: np.ndarray) -> None:
    """Execute one chunk while a dedicated WebSocket connection prefetches the next one."""
    period = 1.0 / args.control_fps
    horizon = args.open_loop_horizon
    worker = AsyncPolicyWorker(args.host, args.port)
    worker.start()
    action_index = 0
    sent = 0
    next_actions = None
    previous_target = None
    boundary_wait_started = None

    try:
        while sent < args.steps:
            loop_start = time.perf_counter()

            completed = worker.poll()
            if completed is not None:
                if completed.error is not None:
                    raise RuntimeError("Background policy inference failed") from completed.error
                next_actions = process_inference_result(completed.result, args)
                logging.info(
                    "Async chunk ready in %.3fs (approximately %.1f control periods)",
                    completed.latency_s,
                    completed.latency_s * args.control_fps,
                )

            remaining = horizon - action_index
            if next_actions is None and not worker.busy and remaining <= args.async_prefetch_threshold:
                observation = robot.get_observation()
                if worker.submit(make_request(observation, args.prompt)):
                    logging.info("Prefetch started with %d actions remaining", remaining)

            if action_index >= horizon:
                if next_actions is None:
                    if boundary_wait_started is None:
                        boundary_wait_started = time.perf_counter()
                        logging.warning("Async queue underrun; holding the last target while inference finishes")
                    if time.perf_counter() - boundary_wait_started > args.async_result_timeout:
                        raise TimeoutError("Timed out waiting for the next asynchronous action chunk")
                    time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))
                    continue
                actions = blend_chunk(next_actions, previous_target, args.async_transition_steps)
                next_actions = None
                action_index = 0
                boundary_wait_started = None

            target = actions[action_index]
            robot.send_action(dict(zip(ACTION_NAMES, target, strict=True)))
            previous_target = target.copy()
            action_index += 1
            sent += 1
            time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        worker.close()


def run_rtc_continuous(
    robot, args: argparse.Namespace, actions: np.ndarray, *, initial_latency_s: float = 0.0
) -> None:
    """Continuously replace the queue with delay-aligned chunks generated using JAX RTC guidance."""
    period = 1.0 / args.control_fps
    worker = AsyncPolicyWorker(args.host, args.port)
    worker.start()
    actions = actions[: args.open_loop_horizon]
    action_index = 0
    sent = 0
    request_sent_at = None
    estimated_delay = int(np.ceil(initial_latency_s * args.control_fps))
    queue_wait_started = None

    try:
        while sent < args.steps:
            loop_start = time.perf_counter()
            completed = worker.poll()
            if completed is not None:
                if completed.error is not None:
                    raise RuntimeError("Background RTC inference failed") from completed.error
                if request_sent_at is None:
                    raise RuntimeError("Received an RTC result without a matching request")
                actual_delay = sent - request_sent_at
                new_actions = process_inference_result(completed.result, args)[: args.open_loop_horizon]
                if actual_delay >= len(new_actions):
                    raise RuntimeError(
                        f"RTC inference consumed {actual_delay} control periods, but the chunk has only "
                        f"{len(new_actions)} actions; lower FPS or inference latency"
                    )
                actions = new_actions[actual_delay:]
                action_index = 0
                request_sent_at = None
                queue_wait_started = None
                estimated_delay = int(np.ceil(completed.latency_s * args.control_fps))
                logging.info(
                    "RTC chunk merged: latency=%.3fs estimated_delay=%d actual_delay=%d retained=%d",
                    completed.latency_s,
                    estimated_delay,
                    actual_delay,
                    len(actions),
                )
                if estimated_delay >= args.rtc_execution_horizon:
                    logging.warning(
                        "RTC delay %d reaches/exceeds execution horizon %d; increase --rtc-execution-horizon",
                        estimated_delay,
                        args.rtc_execution_horizon,
                    )

            remaining = len(actions) - action_index
            if request_sent_at is None and not worker.busy and 0 < remaining <= args.rtc_queue_threshold:
                if remaining < args.rtc_execution_horizon:
                    logging.warning(
                        "Only %d old actions remain for an RTC execution horizon of %d",
                        remaining,
                        args.rtc_execution_horizon,
                    )
                observation = robot.get_observation()
                prefix = actions[action_index : action_index + args.rtc_execution_horizon]
                request = make_rtc_request(
                    observation,
                    args.prompt,
                    prefix,
                    inference_delay=min(estimated_delay, args.open_loop_horizon),
                    execution_horizon=min(args.rtc_execution_horizon, len(prefix)),
                    max_guidance_weight=args.rtc_max_guidance_weight,
                )
                if worker.submit(request):
                    request_sent_at = sent
                    logging.info(
                        "RTC inference started: remaining=%d estimated_delay=%d prefix=%d",
                        remaining,
                        estimated_delay,
                        len(prefix),
                    )

            if action_index >= len(actions):
                if queue_wait_started is None:
                    queue_wait_started = time.perf_counter()
                    logging.warning("RTC queue underrun; holding the last target")
                if time.perf_counter() - queue_wait_started > args.async_result_timeout:
                    raise TimeoutError("Timed out waiting for the next RTC action chunk")
                time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))
                continue

            target = actions[action_index]
            robot.send_action(dict(zip(ACTION_NAMES, target, strict=True)))
            action_index += 1
            sent += 1
            time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        worker.close()


def main() -> None:
    args = parse_args()
    if args.open_loop_horizon < 1 or args.open_loop_horizon > 50:
        raise ValueError("--open-loop-horizon must be in [1, 50]")
    if args.async_prefetch_threshold is None:
        args.async_prefetch_threshold = min(40, args.open_loop_horizon)
    if args.execute and args.confirm_motion != "YES":
        raise ValueError("Motion requires both --execute and --confirm-motion=YES")
    if args.continuous and not args.execute:
        raise ValueError("--continuous requires --execute")
    if args.continuous and args.confirm_continuous != "YES":
        raise ValueError("Continuous motion requires --continuous --confirm-continuous=YES")
    if args.async_mode and not args.continuous:
        raise ValueError("--async requires --continuous")
    if args.rtc and not args.continuous:
        raise ValueError("--rtc requires --continuous")
    if args.rtc and args.async_mode:
        raise ValueError("Choose either --rtc or --async, not both")
    if not 1 <= args.async_prefetch_threshold <= args.open_loop_horizon:
        raise ValueError("--async-prefetch-threshold must be in [1, open-loop-horizon]")
    if not 0 <= args.async_transition_steps <= args.open_loop_horizon:
        raise ValueError("--async-transition-steps must be in [0, open-loop-horizon]")
    if args.async_result_timeout <= 0:
        raise ValueError("--async-result-timeout must be positive")
    if args.rtc:
        if not 1 <= args.rtc_execution_horizon <= args.open_loop_horizon:
            raise ValueError("--rtc-execution-horizon must be in [1, open-loop-horizon]")
        if not args.rtc_execution_horizon <= args.rtc_queue_threshold <= args.open_loop_horizon:
            raise ValueError("--rtc-queue-threshold must be in [rtc-execution-horizon, open-loop-horizon]")
        if args.rtc_max_guidance_weight <= 0:
            raise ValueError("--rtc-max-guidance-weight must be positive")
    if args.execute and args.max_relative_target <= 0:
        raise ValueError("Motion requires a positive --max-relative-target")

    cameras = {
        "wrist": OpenCVCameraConfig(
            index_or_path=camera_source(args.wrist_camera), width=640, height=480, fps=args.camera_fps
        ),
        "front": OpenCVCameraConfig(
            index_or_path=camera_source(args.front_camera), width=640, height=480, fps=args.camera_fps
        ),
    }
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir.expanduser(),
            cameras=cameras,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=True,
        )
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Policy server metadata: %s", client.get_server_metadata())

    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("SO-101 is not calibrated; refusing inference or motion")

        observation = robot.get_observation()
        actions = infer_actions(client, observation, args)
        print("Current state:", make_request(observation, args.prompt)["observation.state"])
        print("Predicted chunk shape:", actions.shape)
        print("First five targets:\n", actions[:5])
        print("Policy range:", actions.min(axis=0), actions.max(axis=0))
        if not args.execute:
            print("Dry run complete; no motor action was sent. Add --execute --confirm-motion=YES only after review.")
            return

        if args.continuous:
            if args.rtc:
                logging.info("Warming up the JAX RTC graph before enabling motor execution")
                warmup_started = time.perf_counter()
                warmup_request = make_rtc_request(
                    observation,
                    args.prompt,
                    actions[: args.rtc_execution_horizon],
                    inference_delay=0,
                    execution_horizon=args.rtc_execution_horizon,
                    max_guidance_weight=args.rtc_max_guidance_weight,
                )
                process_inference_result(client.infer(warmup_request), args)
                logging.info("JAX RTC compile warmup complete in %.3fs", time.perf_counter() - warmup_started)
                hot_started = time.perf_counter()
                process_inference_result(client.infer(warmup_request), args)
                hot_latency_s = time.perf_counter() - hot_started
                logging.info(
                    "JAX RTC hot warmup complete in %.3fs (estimated delay=%d)",
                    hot_latency_s,
                    int(np.ceil(hot_latency_s * args.control_fps)),
                )
                run_rtc_continuous(robot, args, actions, initial_latency_s=hot_latency_s)
            elif args.async_mode:
                run_async_continuous(robot, args, actions)
            else:
                run_continuous(robot, client, args, actions)
        else:
            print("Entering default step mode: review each target, then press s to send and advance.")
            run_step_mode(robot, client, args)
    except KeyboardInterrupt:
        logging.info("Interrupted; disconnecting with torque disable")
    finally:
        with contextlib.suppress(Exception):
            robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
