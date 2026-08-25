#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HIPHI_ROOT="${HIPHI_ROOT:-$HOME/Chris/HiPHI}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/hsretargeting/bin/python}"

MANIFEST_DIR="$REPO_ROOT/pipelines/hiphi_to_g1/manifests"
SMOKE_DIR="$HIPHI_ROOT/validation_smokes"
RETARGET_DIR="$HIPHI_ROOT/retarget_225_hiphi"
CONVERTED_DIR="$HIPHI_ROOT/converted_30cm_124"

case "${1:-help}" in
prepare)
    export HIPHI_ROOT
    "$PYTHON_BIN" "$REPO_ROOT/scripts/prepare_hiphi_validation_hpc.py"
    ;;

retarget)
    mkdir -p "$RETARGET_DIR"
    TASK_LIST="$HIPHI_ROOT/hiphi_retarget_tasks.txt"

    find "$SMOKE_DIR" -maxdepth 1 -name '*_pickup_smoke.npz' \
        -printf '%f\n' | sed 's/\.npz$//' | sort > "$TASK_LIST"

    N=$(wc -l < "$TASK_LIST")

    if (( N == 0 )); then
        echo "No smoke motions found"
        exit 1
    fi

    export REPO_ROOT HIPHI_ROOT TASK_LIST
    export RETARGET_PYTHON="$PYTHON_BIN"

    cd "$REPO_ROOT"
    sbatch --array="0-$((N-1))%32" retarget_225_hiphi.sbatch
    ;;

contact-qc)
    export HIPHI_ROOT
    "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_hiphi_smokes.py"
    ;;

body-qc)
    export HIPHI_ROOT
    "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_fullbody_box_penetration.py"
    ;;

convert-30cm)
    mkdir -p "$CONVERTED_DIR"

    while IFS= read -r task; do
        [[ -z "$task" ]] && continue

        input="$RETARGET_DIR/${task}_original.npz"
        output="$CONVERTED_DIR/${task}_mj_w_obj.npz"

        if [[ ! -f "$input" ]]; then
            echo "[MISSING] $input"
            continue
        fi

        if [[ -f "$output" ]]; then
            echo "[SKIP] $task"
            continue
        fi

        echo "[CONVERT] $task"

        "$PYTHON_BIN"             "$REPO_ROOT/src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py"             --input-file "$input"             --output-name "$output"             --robot g1             --object-name box_0p3000_0p3000_0p3000             --input-fps 30             --output-fps 50             --has-dynamic-object             --no-use-omniretarget-data             --once

    done < "$MANIFEST_DIR/usable_30cm_124.txt"
    ;;

status)
    echo "Smoke:"
    find "$SMOKE_DIR" -maxdepth 1 -name '*_pickup_smoke.npz' 2>/dev/null | wc -l

    echo "Retargeted:"
    find "$RETARGET_DIR" -maxdepth 1 -name '*_original.npz' 2>/dev/null | wc -l

    echo "Converted:"
    find "$CONVERTED_DIR" -maxdepth 1 -name '*_mj_w_obj.npz' 2>/dev/null | wc -l
    ;;

*)
    echo "Usage: $0 {prepare|retarget|contact-qc|body-qc|convert-30cm|status}"
    ;;
esac
