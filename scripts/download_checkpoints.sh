#!/usr/bin/env bash
# Download pre-trained Carry2Anywhere checkpoints from Hugging Face.
#
# Usage: bash scripts/download_checkpoints.sh
#
# Layout after running:
#   checkpoints/
#   ├── Teacher/model_177999.pt
#   └── Student/
#       ├── model_14000.pt
#       └── model_20000.pt

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
CKPT_DIR="${ROOT_DIR}/checkpoints"
HF_REPO="yeager1225/Carry2Anywhere"

mkdir -p "${CKPT_DIR}/Teacher" "${CKPT_DIR}/Student"

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "[*] Downloading via huggingface-cli from ${HF_REPO} ..."
  huggingface-cli download "${HF_REPO}" \
    --local-dir "${CKPT_DIR}" \
    --local-dir-use-symlinks False
  echo "[+] Done. Checkpoints are under ${CKPT_DIR}/"
  exit 0
fi

echo "[!] huggingface-cli not found, falling back to direct HTTPS download."
echo "    To get progress bars and resume support, run:"
echo "        pip install -U huggingface_hub"
echo "    and re-run this script."
echo

BASE="https://huggingface.co/${HF_REPO}/resolve/main"
declare -a FILES=(
  "Teacher/model_177999.pt"
  "Student/model_14000.pt"
  "Student/model_20000.pt"
)

for rel in "${FILES[@]}"; do
  out="${CKPT_DIR}/${rel}"
  if [[ -f "${out}" ]]; then
    echo "[=] ${rel} already exists, skipping."
    continue
  fi
  echo "[*] ${rel}"
  curl -L --fail --progress-bar -o "${out}.tmp" "${BASE}/${rel}"
  mv "${out}.tmp" "${out}"
done

echo "[+] Done. Checkpoints are under ${CKPT_DIR}/"
