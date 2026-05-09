# Carry2Anywhere · Environment Setup

This document is the minimum reproducible setup for **retargeting + simulation
training**. It is the long-form companion to the *Getting Started* section of
the [README](README.md).

The matching install scripts live under `scripts/`:

- `scripts/setup_retargeting.sh` — installs the `hsretargeting` env (data prep)
- `scripts/setup_isaacsim.sh` — installs the `hssim` env (Isaac Sim training)

Deployment-only scripts (`setup_inference.sh`, `setup_isaacgym.sh`,
`setup_mujoco.sh`) are **not required** for training/distillation and are not
covered here.

---

## 1. Hardware / OS requirements

| Item            | Verified                              | Notes                                                         |
| --------------- | ------------------------------------- | ------------------------------------------------------------- |
| OS              | Ubuntu 22.04 LTS                      | Isaac Sim is Linux-only.                                      |
| CPU             | x86_64                                |                                                               |
| GPU             | NVIDIA RTX 3090 (24 GB)               | Any RTX with ≥ 16 GB; more VRAM lets you scale `num_envs` up. |
| NVIDIA driver   | 580.142 (verified)                    | Need ≥ 535 for Isaac Sim 5.1 + cu128.                         |
| Free disk       | ~30 GB                                | Conda envs + Isaac Sim cache are large.                       |
| Network         | PyPI / pypi.nvidia.com / GitHub       | Downloads ~10 GB on first install.                            |

Sanity check the driver:

```bash
nvidia-smi
```

---

## 2. Directory layout

The setup scripts default to `~/.holosoma_deps/miniconda3` for the conda root.
If you already have a miniconda installation, just symlink it in (no need to
reinstall):

```bash
mkdir -p ~/.holosoma_deps
ln -sfn ~/miniconda3 ~/.holosoma_deps/miniconda3
```

On a fresh machine the scripts download miniconda automatically.

Two conda envs will be created:

| Env name        | Python | Purpose                                   |
| --------------- | ------ | ----------------------------------------- |
| `hsretargeting` | 3.11   | Human-motion retargeting + data preparation. |
| `hssim`         | 3.11   | Isaac Sim / Isaac Lab training & evaluation. |

---

## 3. System packages

```bash
sudo apt install -y curl unzip git cmake build-essential
```

Ubuntu 22.04 normally ships with all of these. `setup_isaacsim.sh` only calls
`sudo apt` if `cmake` or `gcc` is actually missing, so it is safe to run on
existing machines.

---

## 4. Install the retargeting env (`hsretargeting`)

```bash
cd <repo-root>
bash scripts/setup_retargeting.sh
```

The script will:

1. Install miniconda into `~/.holosoma_deps/miniconda3` (skipped if already present).
2. Install `mamba` into `base` (first run only).
3. `mamba create -y -n hsretargeting python=3.11 -c conda-forge --override-channels`.
4. `pip install -e src/holosoma_retargeting` (numpy 2.x, torch, mujoco, yourdfpy, viser, smplx, cvxpy, libigl, …).
5. Touch a sentinel file `~/.holosoma_deps/.env_setup_retargeting_hsretargeting`.

Smoke test:

```bash
source scripts/source_retargeting_setup.sh
python -c "import torch, mujoco, holosoma_retargeting; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('mujoco', mujoco.__version__)"
```

> The PyPI package is `libigl` but the import name is `igl`:
>
> ```python
> import igl
> ```

---

## 5. Install the training env (`hssim`, Isaac Sim 5.1 + Isaac Lab v2.3.0)

```bash
cd <repo-root>
bash scripts/setup_isaacsim.sh
```

> Downloads ~7–10 GB and takes 30–60 minutes the first time. The
> `pypi.nvidia.com` index cannot be mirrored; PyPI itself can be accelerated
> via `pip config set global.index-url <mirror>`.

The script handles:

1. Reusing / creating miniconda.
2. `mamba create -y -n hssim python=3.11 -c conda-forge --override-channels`.
3. `conda install -c conda-forge -y ffmpeg libiconv libglu`.
4. `pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`.
5. `pip install pyperclip`.
6. `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`.
7. `git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 ~/.holosoma_deps/IsaacLab`.
8. (only if missing) `sudo apt install -y cmake build-essential`.
9. `./isaaclab.sh --install` with the following workarounds baked in:
   - `CMAKE_POLICY_VERSION_MINIMUM=3.5` to bypass `egl_probe`'s cmake compatibility issue.
   - `TERM=xterm-256color` so `tabs 4` works under non-interactive shells.
   - `OMNI_KIT_ACCEPT_EULA=1`, `PRIVACY_CONSENT=Y` to auto-accept the Omniverse EULA.
   - `PIP_CONSTRAINT=setuptools<82` so `flatdict==4.0.1` (sdist-only, depends on `pkg_resources`) can build.
10. Explicit `pip install -e ~/.holosoma_deps/IsaacLab/source/isaaclab` because
    `--install` uses `find ... -exec` which silently swallows failures and
    occasionally skips the core `isaaclab` package.
11. `pip install -e src/holosoma[unitree,booster]`.
12. `pip install --upgrade 'wandb>=0.21.1'` (override `rl-games`'s pin).
13. `pip install 'numpy==1.23.5' 'setuptools<82'` — `holosoma` pins
    `numpy==1.23.5`, but the Isaac Lab install bumps it to 1.26.x, so we
    restore it.
14. Touch a sentinel file `~/.holosoma_deps/.env_setup_finished_hssim`.

> You will see a flurry of `dependency conflicts` warnings during install (e.g.
> `isaacsim-kernel 5.1.0.0 requires numpy==1.26.0, but you have numpy
> 1.23.5`). These are tolerated — training runs fine in our experiments.

Smoke test:

```bash
source scripts/source_isaacsim_setup.sh   # activates hssim + exports OMNI_KIT_ACCEPT_EULA=1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0))"
python src/holosoma/holosoma/train_agent.py --help
```

A deeper import check (first launch takes ~80 s for shader compilation):

```bash
source scripts/source_isaacsim_setup.sh
python - <<'PY'
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
import isaaclab, isaaclab.envs, isaaclab_tasks, holosoma
print("isaaclab:", isaaclab.__file__)
print("OK")
sim.close()
PY
```

---

## 6. First training / eval run

Always activate `hssim` first:

```bash
source scripts/source_isaacsim_setup.sh
```

### 6.1 Train the teacher (G1 29-DoF whole-body tracking with object)

```bash
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-wbt-w-object \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=src/holosoma/holosoma/motions \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_w_obj.npz" \
  --training.num_envs=4096
```

Resume from a checkpoint with `--training.checkpoint <path>.pt`.

### 6.2 Evaluate (headless, 4 envs)

```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint checkpoints/Teacher/model_177999.pt \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=src/holosoma/holosoma/motions \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_w_obj.npz" \
  --command.setup_terms.motion_command.params.motion_config.eval_motion_id=-1 \
  --training.headless=True --training.num_envs=4 \
  --simulator.config.scene.env_spacing=5.0
```

For the full distillation pipeline see the README.

---

## 7. Common issues

### 7.1 `import libigl` raises `ModuleNotFoundError`

Use `import igl` (PyPI name vs import name mismatch).

### 7.2 IsaacLab install: `tabs: terminal type 'dumb' cannot reset tabs`

Run with `TERM=xterm-256color ./isaaclab.sh --install`. The setup script
already does this.

### 7.3 IsaacLab install: `Unable to bootstrap inner kit kernel: EOF when reading a line`

The Omniverse EULA prompt is being asked. Export both:

```bash
export OMNI_KIT_ACCEPT_EULA=1
export PRIVACY_CONSENT=Y
```

### 7.4 IsaacLab install: `ModuleNotFoundError: No module named 'pkg_resources'` / `Failed to build flatdict`

`setuptools` 82+ removed `pkg_resources`, but `flatdict==4.0.1` (sdist) needs
it at build time. Pin setuptools in pip's build-isolation env:

```bash
echo "setuptools<82" > /tmp/pip-constraints.txt
PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install ...
```

### 7.5 `import isaaclab` fails after install

`./isaaclab.sh --install` silently skips broken sub-installs. Reinstall the
core package explicitly:

```bash
source scripts/source_isaacsim_setup.sh
echo "setuptools<82" > /tmp/pip-constraints.txt
PIP_CONSTRAINT=/tmp/pip-constraints.txt \
  pip install -e ~/.holosoma_deps/IsaacLab/source/isaaclab
```

### 7.6 `numpy` version conflicts

In the env:
- `holosoma` pins `numpy==1.23.5`,
- `isaaclab --install` bumps it to `~1.26`,
- `isaacsim-kernel` declares `numpy==1.26.0`.

In practice training runs fine on `numpy==1.23.5`. The setup script restores
it at the end. If a future Isaac Sim release hard-requires 1.26, relax the
pin in `src/holosoma/pyproject.toml`.

### 7.7 Reset the `hssim` env

```bash
bash scripts/reset_isaacsim.sh   # asks for confirmation, removes hssim env + IsaacLab + sentinel
bash scripts/setup_isaacsim.sh
```

For `hsretargeting`:

```bash
~/.holosoma_deps/miniconda3/bin/conda env remove -y -n hsretargeting
rm -f ~/.holosoma_deps/.env_setup_retargeting_hsretargeting
bash scripts/setup_retargeting.sh
```

### 7.8 Disable Weights & Biases

W&B logging is **off by default**. To enable, append `logger:wandb` to your
training command and run `wandb login` once.

---

## 8. Verified versions

| Component         | Version                  |
| ----------------- | ------------------------ |
| OS                | Ubuntu 22.04.5 LTS       |
| Kernel            | 6.8.0-111-generic        |
| GPU               | RTX 3090                 |
| NVIDIA driver     | 580.142                  |
| Python (both envs)| 3.11.15                  |
| **hsretargeting** |                          |
| numpy             | 2.3.5                    |
| torch             | 2.11.0+cu130             |
| mujoco            | 3.8.0                    |
| **hssim**         |                          |
| numpy             | 1.23.5                   |
| torch             | 2.7.0+cu128              |
| isaacsim          | 5.1.0.0                  |
| isaaclab          | 0.47.2 (IsaacLab v2.3.0) |
| holosoma          | 0.0.1 (editable)         |
| wandb             | 0.26.1                   |
