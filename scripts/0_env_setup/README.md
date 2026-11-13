# 0 — Environment setup

> **Cloned verbatim from `slavic-speech-pipeline/0_env_setup/` on 2026-11-13.**
> Env names (`ssp`, `ssp-cuda`) are intentionally shared with SSP so one env
> satisfies both repos. If the SSP envs already exist on your machine, you can
> skip the setup scripts and just `mamba activate ssp` / `ssp-cuda`.

One command per environment, driven by a frozen YAML. The YAML is exported
from a known-good machine and lives next to this file. No requirements
files, no per-package hand-holding, no drift.

## Run

**CPU** (data prep, chapter 1, light development):

```bash
bash scripts/0_env_setup/setup_env_cpu.sh
mamba activate ssp
```

**CUDA** (GPU training):

```bash
bash scripts/0_env_setup/setup_env_cuda.sh
mamba activate ssp-cuda
```

Add `--force` to either script to remove and recreate an existing env.

## Prerequisites

`mamba` on PATH. The recommended tool is **Miniforge**, which bundles
mamba by default. Install it once:

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p "${HOME}/miniforge3"
```

Then activate `mamba` in every new shell by adding these two lines to
`~/.bashrc`:

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
```

**Alternative — micromamba.** `micromamba` is a separate binary from
`mamba` with a compatible command set. Alias it if that is what the host
provides:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

See `ENV_SETUP.md` for the guarded `.bashrc` form and troubleshooting.

## Verify

After activation:

```bash
python -c "import torch, datasets, soundfile; print(torch.__version__, datasets.__version__)"
```

CPU: `torch.cuda.is_available()` returns `False`. CUDA: `torch.cuda.is_available()`
returns `True`. A `False` on the CUDA env means the CUDA build did not take.

## Exporting a new YAML

When the env changes and the new state should be shared:

```bash
mamba env export -n ssp      --no-builds > scripts/0_env_setup/ssp_cpu.yaml
mamba env export -n ssp-cuda --no-builds > scripts/0_env_setup/ssp_cuda.yaml
```

`--no-builds` drops per-machine build hashes, so the YAML replays on other
Linux boxes.

The CUDA export drops the `--extra-index-url` line from the pip section.
Re-add it manually above the tagged torch line after every re-export:

```yaml
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu124
    - torch==2.6.0+cu124
```

## Backups

Per-machine YAML backups live under `backups/env_backup_<host>/`. Each
host folder has three exports: `<env>.yaml` (portable), `<env>_with_builds.yaml`
(exact), and `<env>_from_history.yaml` (intent). The canonical `ssp_cpu.yaml`
and `ssp_cuda.yaml` at the top of `scripts/0_env_setup/` are the current source of
truth. Backups are for cross-machine diffing if the canonical YAML ever
regresses.

## Windows users

The scripts here are `bash` scripts. On Windows, run them under **Git Bash**
(from Git for Windows), not PowerShell or CMD. The rest of the pipeline is
portable Python and runs the same way.

> Developed and tested on Linux. The steps below are what a Windows user
> reported worked for them. Treat as an unverified starting point rather
> than official support.

Do this once before running `setup_env_cpu.sh`:

1. **Enable long file paths.** Mamba envs regularly exceed the Windows
   260-character path limit. Open PowerShell as Administrator and run:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
2. **Put Miniforge on your Git Bash PATH.** Add these lines to `~/.bashrc`:
   ```bash
   export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/Library/bin:$PATH"
   source "$HOME/miniforge3/etc/profile.d/conda.sh"
   source "$HOME/miniforge3/etc/profile.d/mamba.sh"
   ```
3. **Run Git Bash as Administrator** for the setup script. Env creation
   writes many files, and UAC otherwise blocks some of the writes.

If `setup_env_cpu.sh` still cannot find mamba once you are in Git Bash,
run the PATH export from step 2 in the current shell and try again.

## `legacy/`

Holds the pre-YAML pair of `setup_env_*.sh` and `requirements_*.txt` files.
Gitignored, kept locally for reference. Not the intended install path
anymore.
