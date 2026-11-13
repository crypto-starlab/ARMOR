#!/usr/bin/env bash
# One terminal per role; config is a trusted Bash file, never a received payload.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || $# -lt 2 ]]; then
    echo 'Usage: bash role.sh lan|wan keycenter|server2|server1|client [--dry-run] [Python options]'
    echo 'Start roles in the order shown, using the same ARMOR_AUTHKEY and RUN_TAG.'
    exit 0
fi
profile=$1
role=$2
shift 2
case "$profile" in lan|wan) ;; *) echo 'Expected lan or wan' >&2; exit 2 ;; esac
case "$role" in keycenter|server1|server2|client) ;; *) echo 'Unknown role' >&2; exit 2 ;; esac
source "$script_dir/config.env"
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
python_bin=${ARMOR_PYTHON:-$(command -v python)}
python_bin=$(command -v -- "$python_bin")
keycenter_address=$KEYCENTER_HOST
server1_address=$SERVER1_HOST
server2_address=$SERVER2_HOST
case "$role" in
    keycenter) keycenter_address=$KEYCENTER_BIND; gpu=$KEYCENTER_GPU ;;
    server1) server1_address=$SERVER1_BIND; gpu=$SERVER1_GPU ;;
    server2) server2_address=$SERVER2_BIND; gpu=$SERVER2_GPU ;;
    client) gpu=$CLIENT_GPU ;;
esac
armor_dir=$(cd -- "$script_dir/.." && pwd)
export PYTHONPATH="$armor_dir/phantom-fhe/build/lib:$armor_dir${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$armor_dir/phantom-fhe/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
command_args=("$python_bin" "$script_dir/$role.py"
    --gpu "$gpu" --keycenter_host "$keycenter_address" --keycenter_port "$KEYCENTER_PORT"
    --server1_host "$server1_address" --server1_port "$SERVER1_PORT"
    --server2_host "$server2_address" --server2_port "$SERVER2_PORT"
    --connect_timeout "$CONNECT_TIMEOUT")
if [[ $role == keycenter ]]; then
    command_args+=(--dataset "$DATASET" --method armor --num_clients "$NUM_CLIENTS"
        --global_ep "$GLOBAL_EP" --local_ep "$LOCAL_EP" --local_bs "$LOCAL_BS"
        --attack "$ATTACK" --attack_ratio "$ATTACK_RATIO" --attack_time "$ATTACK_TIME"
 --seed "$SEED" --extract_partial_constant
        --log_path "logs/${RUN_TAG}_{role}.jsonl")
fi
command_args+=("$@")
namespace=
if [[ $role == server2 ]]; then namespace=armor-s2; fi
if [[ $role == client ]]; then namespace=armor-client; fi
if [[ -n $namespace ]]; then
    command_args=(sudo --preserve-env=ARMOR_AUTHKEY ip netns exec "$namespace"
        env "PYTHONPATH=$PYTHONPATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        PYTHONDONTWRITEBYTECODE=1 "${command_args[@]}")
fi
if $dry_run; then
    printf 'profile=%s role=%s\n' "$profile" "$role"
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
fi
: "${ARMOR_AUTHKEY:?Set the same ARMOR_AUTHKEY in all four terminals}"
cd -- "$script_dir"
if [[ $role == keycenter ]]; then
    [[ ! -e logs/${RUN_TAG}_keycenter.jsonl ]] || { echo "Choose a fresh RUN_TAG: $RUN_TAG" >&2; exit 1; }
fi
exec "${command_args[@]}"
