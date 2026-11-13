#!/usr/bin/env bash
# Repeat both dimensions; publish each successful run and archive its old results.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
    echo 'Usage: bash rerun_ext_armor.sh [--dry-run]' >&2
    exit 2
fi
dry_run=${1:-}
if [[ -z $dry_run ]]; then sudo -v; fi
stamp=$(date +%Y%m%d_%H%M%S)_$$
for ringdim in 32768 65536; do
    for method in wo_Ext ARMOR; do
        tag="rerun_${ringdim}_${stamp}"
        bash "$script_dir/$method/lan.sh" --ringdim "$ringdim" --run-tag "$tag" ${dry_run:+--dry-run}
        if [[ -n $dry_run ]]; then continue; fi
        # The runner returns success only after validation and CSV/PNG/PDF creation.
        exec 9>/tmp/armor_fig4_lan.lock
        flock -n 9 || { echo "Another Fig4 run started; fresh results retained in $tag" >&2; exit 1; }
        folder="$script_dir/$method/results_$ringdim"
        if [[ -e $folder ]]; then
            mkdir -p -- "$script_dir/$method/archive"
            mv -- "$folder" "$script_dir/$method/archive/results_${ringdim}_${stamp}"
        fi
        mv -- "$script_dir/$method/$tag" "$folder"
        flock -u 9
        exec 9>&-
        echo "Updated: $folder"
    done
done
