# Environment setup — details

> **Cloned verbatim from `slavic-speech-pipeline/0_env_setup/ENV_SETUP.md` on 2026-11-13.**
> Env names (`ssp`, `ssp-cuda`) are shared with SSP on purpose — one env, two repos.

Two environment variants: **CPU** (development, laptops) and **CUDA** (GPU
servers). Both are cloned from a frozen `mamba env export` YAML — no
per-package pip lines to keep in sync.

## Prerequisites

`mamba` on PATH. The recommended route is **Miniforge**, which bundles mamba
and conda-forge by default:

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p "${HOME}/miniforge3"
```

To make `mamba activate` work in every new shell, add these two lines to
`~/.bashrc` (or `~/.zshrc`):

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
```

Guard the sources if you use the same `.bashrc` on multiple machines:

```bash
# Guarded form — silently skips on machines without Miniforge.
if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    if [ -f "${HOME}/miniforge3/etc/profile.d/mamba.sh" ]; then
        source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
    fi
fi
```

Then `source ~/.bashrc` or open a new terminal.

> **Note:** using only `conda shell.bash hook` wires up `conda activate` but
> not `mamba activate`. Sourcing both `profile.d` files above wires up
> both, so either works.

### If your host only has micromamba

`micromamba` is a separate binary from `mamba` with a compatible command
set. The scripts here call `mamba`, so alias it:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

Add both lines to `~/.bashrc` for persistence.

---

## CPU environment (`ssp`)

For local development and data prep. No GPU required.

```bash
bash scripts/0_env_setup/setup_env_cpu.sh          # create, or bail if `ssp` already exists
bash scripts/0_env_setup/setup_env_cpu.sh --force  # remove and recreate

mamba activate ssp
```

The script is a thin wrapper around:

```bash
mamba env create -n ssp -f scripts/0_env_setup/ssp_cpu.yaml
```

`ssp_cpu.yaml` was exported with `--no-builds` from a known-good machine
and is the single source of truth for the env. PyTorch is installed
through pip as `torch==2.12.0+cpu`, so the env contains no CUDA runtime
libraries.

---

## CUDA environment (`ssp-cuda`)

For GPU servers. Targets **CUDA 12.4**, compatible with driver 550 or newer.

```bash
bash scripts/0_env_setup/setup_env_cuda.sh          # create, or bail if `ssp-cuda` already exists
bash scripts/0_env_setup/setup_env_cuda.sh --force  # remove and recreate

mamba activate ssp-cuda
```

Thin wrapper around:

```bash
mamba env create -n ssp-cuda -f scripts/0_env_setup/ssp_cuda.yaml
```

`ssp_cuda.yaml` was exported with `--no-builds` from a known-good GPU
machine. PyTorch is installed through pip as `torch==2.6.0+cu124` and
`torchaudio==2.6.0+cu124`. The YAML has a top-of-`pip:` line
`--extra-index-url https://download.pytorch.org/whl/cu124`, without which
pip cannot resolve these tagged wheels during replay.

Verify GPU is visible after activation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Starting fresh

```bash
mamba env remove -n ssp        # CPU
mamba env remove -n ssp-cuda   # CUDA
```

Or `bash scripts/0_env_setup/setup_env_cpu.sh --force` for the CPU env — one step.

---

## Exporting a new YAML

Whenever the env changes and the new state should be shared:

```bash
mamba env export -n ssp      --no-builds > scripts/0_env_setup/ssp_cpu.yaml
mamba env export -n ssp-cuda --no-builds > scripts/0_env_setup/ssp_cuda.yaml
```

`--no-builds` drops per-machine build hashes (`numpy=1.26.4=py311hXXXX` becomes
`numpy=1.26.4`), so the YAML replays on other Linux boxes without hitting
"no such build" errors.

For torch specifically, `mamba env export` drops the `--extra-index-url`
line from the pip section. Re-add it manually above the tagged torch line
after every re-export:

```yaml
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu124
    - torch==2.6.0+cu124
```

---

## Backups

Per-machine backup exports live in `scripts/0_env_setup/backups/env_backup_<host>/`.
Each host folder contains three exports:

- `<env>.yaml` — `--no-builds`, portable across Linux boxes.
- `<env>_with_builds.yaml` — full snapshot with build hashes, exact
  replay on the same platform.
- `<env>_from_history.yaml` — record of explicitly-installed packages.

The canonical `ssp_cpu.yaml` and `ssp_cuda.yaml` at the top of
`scripts/0_env_setup/` are the current single source of truth. Backups exist to
allow cross-machine diffing if the canonical YAML ever regresses.

---

## Notes

- `ssp` and `ssp-cuda` are intentionally separate envs — do not upgrade one
  into the other.
- The GPU server does not need `nvcc`. PyTorch from the pip wheel bundles
  its own CUDA runtime.
- The Jupyter kernel registration is handled by mamba's env resolver
  through the `ipykernel` package in the YAML. If the "Python (ssp)"
  kernel does not appear in Jupyter, run once:
  `python -m ipykernel install --user --name ssp --display-name "Python (ssp)"`.
