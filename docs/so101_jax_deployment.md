# SO101 deployment for an OpenPI JAX checkpoint

Keep the two dependency environments separate. The OpenPI environment owns JAX, Orbax, normalization, transforms, and
the policy server. The existing `lerobot_hil` environment owns the calibrated SO101, cameras, timing, safety checks, and
the lightweight `openpi-client` WebSocket client. Native `lerobot-rollout --policy.path` cannot load an Orbax checkpoint.

## 1. Wrap the downloaded params without copying them

The downloaded `openpi-so101-pi05-60-13000` directory is a params-only Orbax item. Stock `serve_policy.py` expects a
checkpoint directory containing `params/` and `assets/<asset-id>/norm_stats.json`. Create those paths as symlinks:

```bash
cd /home/larry/openpi-kuavo
.venv/bin/python scripts/prepare_so101_checkpoint.py \
  --params-dir /mnt/pqssd/so101/training_outpus/openpi-so101-pi05-60-13000 \
  --norm-stats assets/pi05_so101_60/so101_grab_blue_pen_60/norm_stats.json \
  --output-dir checkpoints/pi05_so101_60/local_13000
```

## 2. Start the JAX server

This reads a roughly 12 GB on-disk Orbax item and the first request triggers JAX compilation. It does not train or alter the
checkpoint.

First run the mandatory hardware-free gate:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/smoke_test_so101_policy.py \
  --checkpoint=/home/larry/openpi-kuavo/checkpoints/pi05_so101_60/local_13000 \
  --require-cuda
```

Only start the server after this reports `actions shape: (50, 6)` and a finite range.

```bash
cd /home/larry/openpi-kuavo
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python scripts/serve_policy.py \
  --port=8000 \
  --default-prompt='Grab the blue pen and place it into the black box' \
  policy:checkpoint \
  --policy.config=pi05_so101_60 \
  --policy.dir=/home/larry/openpi-kuavo/checkpoints/pi05_so101_60/local_13000
```

## 3. Install only the client into `lerobot_hil`

```bash
conda activate lerobot_hil
cd /home/larry/openpi-kuavo/packages/openpi-client
python -m pip install -e . --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

This package does not install JAX. Keep `/home/larry/lerobot_latest` as the existing editable LeRobot source.

Before connecting hardware, verify the environment boundary while the server is running:

```bash
conda activate lerobot_hil
cd /home/larry/openpi-kuavo
python scripts/smoke_test_so101_remote.py --host=127.0.0.1
```

This decodes one real dataset frame in `lerobot_hil`, transports it through `openpi-client`, and requires a finite
`(50, 6)` response. It does not open cameras or a serial port.

## 4. Mandatory dry run

The checked local benchmark configuration uses follower `/dev/ttyACM1`, calibration ID `so101`, wrist camera 0, and
front camera 2. Verify these device identities after every reboot; camera indices can change.

```bash
conda activate lerobot_hil
cd /home/larry/openpi-kuavo
python scripts/run_so101_openpi_client.py \
  --host=127.0.0.1 \
  --robot-port=/dev/ttyACM1 \
  --robot-id=so101 \
  --wrist-camera=0 \
  --front-camera=2
```

Dry run connects the calibrated arm and cameras, requests one `(50, 6)` chunk, validates finite values, clips it to the
SO101-60 training q01/q99 envelope, prints it, and sends no motor target.

## 5. First guarded motion

Clear the workspace, keep an emergency power cutoff in reach, and begin with a short interactive run:

```bash
python scripts/run_so101_openpi_client.py \
  --host=127.0.0.1 \
  --robot-port=/dev/ttyACM1 \
  --robot-id=so101 \
  --wrist-camera=0 \
  --front-camera=2 \
  --steps=5 \
  --max-relative-target=3 \
  --execute \
  --confirm-motion=YES
```

`--execute` is step mode by default. Before every possible send, the client reads a fresh observation and takes the next
action from the current chunk, clips it to the training q01/q99 envelope, limits every joint to
`max-relative-target` from its current position, and prints current/policy/final/delta columns. The single-letter controls
are `s` to send this one target and advance, `r` to discard the remaining chunk and replan, and `q` to quit. After
`open-loop-horizon` reviewed actions, the client automatically replans from the latest observation.

Continuous execution remains available only after step-mode validation and requires a second explicit confirmation:

```bash
python scripts/run_so101_openpi_client.py ... \
  --open-loop-horizon=5 \
  --execute --confirm-motion=YES \
  --continuous --confirm-continuous=YES
```

To avoid blocking for inference at every chunk boundary, prefetch the next chunk on a dedicated WebSocket connection:

```bash
python scripts/run_so101_openpi_client.py \
  --host=127.0.0.1 --port=8000 \
  --robot-port=/dev/ttyACM1 --robot-id=so101 \
  --wrist-camera=0 --front-camera=2 \
  --control-fps=20 --open-loop-horizon=50 \
  --async --async-prefetch-threshold=40 --async-transition-steps=3 \
  --steps=1000000 --max-relative-target=5 \
  --execute --confirm-motion=YES \
  --continuous --confirm-continuous=YES
```

The control thread executes the current chunk at a fixed rate while a background thread owns a separate WebSocket
client and prepares the next chunk. The first few targets at each boundary are blended with the previous target. If
inference misses the boundary, the client holds the last target and waits instead of inventing actions; it aborts after
`--async-result-timeout` seconds. This is asynchronous chunk fallback, not denoising-level RTC guidance.

For smooth asynchronous replacement, use JAX denoising-level Real-Time Chunking instead of the fallback interpolation:

```bash
python scripts/run_so101_openpi_client.py \
  --host=127.0.0.1 --port=8000 \
  --robot-port=/dev/ttyACM1 --robot-id=so101 \
  --wrist-camera=0 --front-camera=2 \
  --control-fps=30 --open-loop-horizon=50 \
  --rtc --rtc-queue-threshold=30 \
  --rtc-execution-horizon=20 --rtc-max-guidance-weight=10 \
  --steps=200 --max-relative-target=5 \
  --execute --confirm-motion=YES \
  --continuous --confirm-continuous=YES
```

RTC sends the unexecuted absolute-action prefix back to the server. The policy applies the same training normalization,
uses prefix guidance inside every JAX flow-matching denoising step, and drops the part of the returned chunk consumed
during the measured inference delay. Before motor execution, the client runs one compile warmup and one hot timing
request. On the local test GPU,
the first RTC compile took about 9.6 seconds and a hot request took about 0.23 seconds (roughly seven periods at 30 Hz).
If the logged delay approaches `rtc-execution-horizon`, increase that horizon; if it approaches the 50-step model
horizon, real-time execution is not sustainable at the selected FPS.

The policy outputs absolute targets in this exact order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
roll, and gripper. The client uses LeRobot's degree normalization (`use_degrees=True`), matching the dataset range.
