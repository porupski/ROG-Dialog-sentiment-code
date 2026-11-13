#!/usr/bin/env bash
# setup_env_cpu.sh — clone the `ssp` (CPU) environment from the frozen YAML.
#
# Usage:
#   bash scripts/0_env_setup/setup_env_cpu.sh          # create the env if it doesn't exist
#   bash scripts/0_env_setup/setup_env_cpu.sh --force  # remove and recreate
#
# The single source of truth for the env is `ssp_cpu.yaml` next to this
# script, exported from a known-good machine. No requirements.txt, no
# per-package pip install lines — mamba resolves the whole thing from the
# YAML. The script auto-detects mamba, falls back to micromamba if the
# real mamba binary is not on PATH, and honours ${MAMBA_EXE} if set.

set -euo pipefail

ENV_NAME="ssp"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="${HERE}/ssp_cpu.yaml"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# ── Resolve the env manager: mamba > micromamba > $MAMBA_EXE ────────────
# Bash aliases (alias mamba=micromamba) do NOT survive into subshells like
# `bash script.sh`, so we need a real binary on PATH or a full path.
if command -v mamba &>/dev/null; then
    MAMBA_CMD="mamba"
elif command -v micromamba &>/dev/null; then
    MAMBA_CMD="micromamba"
elif [[ -n "${MAMBA_EXE:-}" && -x "${MAMBA_EXE}" ]]; then
    MAMBA_CMD="${MAMBA_EXE}"
else
    echo "❌ Neither mamba nor micromamba found on PATH."
    echo "   Install Miniforge (bundles mamba):"
    echo "     curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh"
    echo "     bash miniforge.sh -b -p \"\${HOME}/miniforge3\""
    echo "     source \"\${HOME}/miniforge3/etc/profile.d/conda.sh\""
    echo "     source \"\${HOME}/miniforge3/etc/profile.d/mamba.sh\""
    echo "   (Add the two source lines to ~/.bashrc for persistence.)"
    echo
    echo "   Or if you already have the micromamba binary, expose it in one of:"
    echo "     export PATH=\"/path/to/dir/containing/micromamba:\$PATH\""
    echo "     export MAMBA_EXE=/full/path/to/micromamba"
    exit 1
fi

echo "→ Using: ${MAMBA_CMD}"

if [[ ! -f "${YAML}" ]]; then
    echo "❌ ${YAML} not found."
    exit 1
fi

if "${MAMBA_CMD}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "→ Removing existing '${ENV_NAME}' env (--force) …"
        "${MAMBA_CMD}" env remove -n "${ENV_NAME}" -y
    else
        echo "❌ Env '${ENV_NAME}' already exists."
        echo "   To recreate: bash scripts/0_env_setup/setup_env_cpu.sh --force"
        exit 1
    fi
fi

echo "→ Creating '${ENV_NAME}' from ${YAML} …"
echo "   Environment creates can take 3 to 10 minutes."
echo "   The script prints a heartbeat every 20 s so the run does not look stuck."

# -y suppresses the interactive "Confirm changes: [Y/n]" prompt on some hosts.
"${MAMBA_CMD}" env create -y -n "${ENV_NAME}" -f "${YAML}" &
CREATE_PID=$!

START_TS=$(date +%s)
while kill -0 "${CREATE_PID}" 2>/dev/null; do
    sleep 20
    if kill -0 "${CREATE_PID}" 2>/dev/null; then
        ELAPSED=$(( $(date +%s) - START_TS ))
        printf "   … still running (%dm %02ds elapsed) …\n" $((ELAPSED / 60)) $((ELAPSED % 60))
    fi
done

# Reap the exit code without tripping `set -e`.
CREATE_RC=0
wait "${CREATE_PID}" || CREATE_RC=$?
if [[ ${CREATE_RC} -ne 0 ]]; then
    echo "❌ env create failed (exit ${CREATE_RC})."
    exit ${CREATE_RC}
fi

echo
echo "✅ Done. To activate:"
echo "   ${MAMBA_CMD} activate ${ENV_NAME}"
