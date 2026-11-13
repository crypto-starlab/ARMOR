#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Override with RINGDIMS="16384 32768 65536" if the baseline is also needed.
read -r -a ringdims <<< "${RINGDIMS:-32768 65536}"
for ringdim in "${ringdims[@]}"; do
    for method in wo_Dist wo_Lazy wo_Coeff wo_Ext ARMOR; do
        bash "$script_dir/$method/lan.sh" --ringdim "$ringdim" "$@"
    done
done
