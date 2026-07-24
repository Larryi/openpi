# Kuavo Pi0.5 JAX open-loop evaluation

This evaluator reads observations and 50-step ground-truth action chunks directly from a local LeRobot v3 dataset. It
does not use ROS, simulation, or a real robot. Predictions are converted back to Kuavo absolute joint targets before
comparison with dataset actions.

For the interactive viewer, run:

```bash
scripts/run_open_loop_kuavo.sh --viewer
```

Open `http://127.0.0.1:8501`. The viewer caches the JAX policy and selected LeRobot episode, so changing frames or action
dimensions does not reload the 12 GB checkpoint. It shows RGB observations, first-step errors, horizon/per-dimension MAE,
one selected action trajectory, inference time, and optional visual perturbations.

The launcher discovers CUDA libraries installed by pip, selects the BFSU Python mirror by default, and runs the OpenPI
virtual environment. First verify the checkpoint and CUDA runtime without decoding video or compiling inference:

```bash
scripts/run_open_loop_kuavo.sh \
  --config-name pi05_kuavo \
  --checkpoint /mnt/pqssd/pretrained/pi05_local_jax/params \
  --require-cuda \
  --load-only
```

Run a minimal TASK1 evaluation after load-only succeeds:

```bash
scripts/run_open_loop_kuavo.sh \
  --config-name pi05_kuavo \
  --checkpoint /mnt/pqssd/pretrained/pi05_local_jax/params \
  --episodes 0 \
  --stride 50 \
  --max-samples 3 \
  --video-backend torchcodec \
  --require-cuda
```

For TASK2, use `--config-name pi05_kuavo_task2`; its dataset root, 16-dimensional action layout, three camera views, and
normalization asset are selected by that config. `--dataset-root` and `--repo-id` can override the configured dataset,
but metadata validation will reject incompatible state/action names or missing cameras.

Each run creates a timestamped directory under `outputs/open_loop/<config>/` containing:

- `summary.json`: MAE/RMSE, horizon and dimension metrics, joint/gripper metrics, range violations, and timing;
- `first_action_errors.csv`: first-step prediction, ground truth, and error for every sampled observation;
- `error_stats.npz`: accumulated absolute/squared errors and per-horizon counts;
- `horizon_mae.png` and `dim_mae.png`: compact diagnostic plots when matplotlib is installed.

The first inference includes JAX compilation and should not be used as steady-state latency. Open-loop scores on training
data are diagnostic only and do not establish closed-loop robot safety or success.
