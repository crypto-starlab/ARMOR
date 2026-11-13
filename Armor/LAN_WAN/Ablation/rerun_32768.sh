#!/usr/bin/env bash
# Preserve previous measurements and repeat only wo_Ext and ARMOR at N=32768.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --dry-run ]]; then
    for method in wo_Ext ARMOR; do
        bash "$script_dir/$method/lan.sh" --ringdim 32768 --dry-run
    done
    exit 0
fi
# Check privileges before moving any previous results.
sudo -v
exec 9>/tmp/armor_fig4_lan.lock
flock -n 9 || { echo 'Another Fig4 experiment is running.' >&2; exit 1; }
stamp=$(date +%Y%m%d_%H%M%S)
for method in wo_Ext ARMOR; do
    folder="$script_dir/$method/results_32768"
    if [[ -e "$folder" ]]; then
        backup="$script_dir/$method/archive/results_32768_${stamp}_$$"
        mkdir -p -- "$(dirname -- "$backup")"
        mv -- "$folder" "$backup"
        echo "Preserved: $backup"
    fi
done
flock -u 9
exec 9>&-
for method in wo_Ext ARMOR; do
    bash "$script_dir/$method/lan.sh" --ringdim 32768 "$@"
done
