#!/usr/bin/env bash
# Run one method/role; reuses the existing host <-> Server2 network namespace.
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
    cat <<'HELP'
Usage: bash run_norm_role.sh METHOD ROLE [--profile local|lan|wan] [--dry-run] [Python options]
METHOD: wo_Dist wo_Lazy wo_Ext wo_Coeff Yasuda ARMOR
ROLE: keycenter server2 server1 client
Profile defaults to local (127.0.0.1, no namespace). LAN/WAN profiles only affect Server2's link.
Apply the matching network_profile.sh setup/profile before using a shaped profile.
All roles require the same ARMOR_AUTHKEY. Set RUN_TAG to a fresh experiment tag.
Example: bash run_norm_role.sh ARMOR keycenter --profile lan --input_blocks 62 184 367 --iterations 5
HELP
}
if [[ ${1:-} == --help || $# -lt 2 ]]; then usage; exit 0; fi
method=$1
role=$2
shift 2
case "$method" in wo_Dist|wo_Lazy|wo_Ext|wo_Coeff|Yasuda|ARMOR) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
case "$role" in keycenter|server1|server2|client) ;; *) echo "Unknown role: $role" >&2; exit 2 ;; esac
profile=local
dry_run=false
extra=()
while (($#)); do
    case "$1" in
        --profile) profile=${2:?Missing profile}; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        *) extra+=("$1"); shift ;;
    esac
done
if [[ $profile == local ]]; then
    KEYCENTER_HOST=127.0.0.1; KEYCENTER_BIND=$KEYCENTER_HOST
    SERVER1_HOST=127.0.0.1; SERVER1_BIND=$SERVER1_HOST
    SERVER2_HOST=127.0.0.1; SERVER2_BIND=$SERVER2_HOST
    SERVER2_NETNS=
    KEYCENTER_PORT=${KEYCENTER_PORT:-9100}; SERVER1_PORT=${SERVER1_PORT:-9101}; SERVER2_PORT=${SERVER2_PORT:-9102}
    KEYCENTER_GPU=${KEYCENTER_GPU:-0}; SERVER1_GPU=${SERVER1_GPU:-0}; SERVER2_GPU=${SERVER2_GPU:-0}; CLIENT_GPU=${CLIENT_GPU:-0}
    CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-600}
    RUN_TAG=${RUN_TAG:-local_rep1}
else
    case "$profile" in lan|wan) ;; *) echo "Unknown profile: $profile" >&2; exit 2 ;; esac
    armor_dir=$(cd -- "$root/../../.." && pwd)
    source "$armor_dir/LAN_WAN/config.env"
    SERVER1_HOST=127.0.0.1; SERVER1_BIND=127.0.0.1
    SERVER2_NETNS=armor-s2
    RUN_TAG=${RUN_TAG:-${profile}_rep1}
fi
keycenter=$KEYCENTER_HOST; server1=$SERVER1_HOST; server2=$SERVER2_HOST
case "$role" in
    keycenter) keycenter=$KEYCENTER_BIND; gpu=$KEYCENTER_GPU ;;
    server1) server1=$SERVER1_BIND; gpu=$SERVER1_GPU ;;
    server2) server2=$SERVER2_BIND; gpu=$SERVER2_GPU ;;
    client) gpu=$CLIENT_GPU ;;
esac
python_bin=${ARMOR_PYTHON:-$(command -v python)}
python_bin=$(command -v -- "$python_bin")
command_args=("$python_bin" "$root/$method/$role.py" --gpu "$gpu"
    --keycenter_host "$keycenter" --keycenter_port "$KEYCENTER_PORT"
    --server1_host "$server1" --server1_port "$SERVER1_PORT"
    --server2_host "$server2" --server2_port "$SERVER2_PORT"
    --connect_timeout "$CONNECT_TIMEOUT" --run_tag "$RUN_TAG" "${extra[@]}")
if [[ $role == server2 && -n $SERVER2_NETNS ]]; then
    armor_dir=$(cd -- "$root/../../.." && pwd)
    command_args=(sudo -n --preserve-env=ARMOR_AUTHKEY ip netns exec "$SERVER2_NETNS"
        env "LD_LIBRARY_PATH=$armor_dir/phantom-fhe/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        "${command_args[@]}")
fi
if $dry_run; then
    printf 'method=%s role=%s profile=%s\n' "$method" "$role" "$profile"
    printf '%q ' "${command_args[@]}"; printf '\n'
    exit 0
fi
: "${ARMOR_AUTHKEY:?Set the same ARMOR_AUTHKEY in all four terminals}"
cd -- "$root"
exec "${command_args[@]}"
