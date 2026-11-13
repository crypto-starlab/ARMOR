#!/usr/bin/env bash
# Isolated host <-> Server2 link. No shaping of the host's physical NIC or lo.
set -Eeuo pipefail
# Manage both isolated experiment links; leave the physical NIC unchanged.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} != --link ]]; then
    case ${1:-} in
        setup|profile) [[ ${2:-} == lan || ${2:-} == wan ]] || { echo 'Usage: bash network.sh setup|profile lan|wan' >&2; exit 2; } ;;
        status|cleanup) ;;
        *) echo 'Usage: bash network.sh setup|profile lan|wan; bash network.sh status|cleanup'; exit 0 ;;
    esac
    for link in server2 client; do
        bash "$script_dir/network.sh" --link "$link" "$@"
    done
    exit 0
fi
link=${2:?Missing link}
shift 2
case "$link" in
    server2) ;;
    client)
        export ARMOR_NET_NET_NS=armor-client
        export ARMOR_NET_HOST_VETH=arm-c1 ARMOR_NET_PEER_VETH=arm-c2
        export ARMOR_NET_HOST_IFB=arm-cifb1 ARMOR_NET_PEER_IFB=arm-cifb2
        export ARMOR_NET_HOST_IP=10.203.78.1 ARMOR_NET_PEER_IP=10.203.78.2
        ;;
    *) exit 2 ;;
esac
NET_NS=${ARMOR_NET_NET_NS:-armor-s2}
HOST_VETH=${ARMOR_NET_HOST_VETH:-arm-v1}
PEER_VETH=${ARMOR_NET_PEER_VETH:-arm-v2}
HOST_IFB=${ARMOR_NET_HOST_IFB:-arm-ifb1}
PEER_IFB=${ARMOR_NET_PEER_IFB:-arm-ifb2}
HOST_IP=${ARMOR_NET_HOST_IP:-10.203.77.1}
PEER_IP=${ARMOR_NET_PEER_IP:-10.203.77.2}

usage() {
    cat <<'HELP'
Usage:
  ./network.sh setup lan|wan
  ./network.sh profile lan|wan
  ./network.sh status
  ./network.sh cleanup

lan        3000 Mbit/s per direction, added RTT 0.2 ms
wan        100 Mbit/s per direction, added RTT 80 ms

Host/KeyCenter: 10.203.77.1; Server2 namespace: armor-s2 (10.203.77.2).
Use on a Linux GPU host with iproute2 and IFB support. Commands require sudo.
Cleanup after stopping experiment processes. setup refuses existing owned names.
HELP
}

if [[ ${1:-help} == help || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $EUID -ne 0 ]]; then
    exec sudo bash "$0" --link "$link" "$@"
fi

profile() {
    local bandwidth delay
    case ${1:-} in
        lan) bandwidth=3000mbit; delay=0.1ms ;;
        wan) bandwidth=100mbit; delay=40ms ;;
        *) usage; return 1 ;;
    esac
    local -a netem_args=(netem limit 10000 delay "$delay")
    if [[ -n $bandwidth ]]; then
        netem_args+=(rate "$bandwidth")
    fi
    # Apply on receiver ingress via IFB, independently for the two directions.
    tc qdisc replace dev "$HOST_IFB" root handle 10: "${netem_args[@]}"
    ip netns exec "$NET_NS" tc qdisc replace dev "$PEER_IFB" root handle 20: "${netem_args[@]}"
    echo "Profile=$1; each direction rate=${bandwidth:-unlimited}, delay=$delay"
}

cleanup() {
    ip netns del "$NET_NS" 2>/dev/null || true
    ip link del "$HOST_VETH" 2>/dev/null || true
    ip link del "$HOST_IFB" 2>/dev/null || true
}

setup() {
    command -v ip tc ethtool awk >/dev/null
    case ${1:-} in lan|wan) ;; *) usage; return 1 ;; esac
    if ip netns list | awk -v name="$NET_NS" '$1 == name {found=1} END {exit !found}'; then
        echo "Namespace $NET_NS exists; use profile or cleanup." >&2
        return 1
    fi
    if ip link show dev "$HOST_VETH" >/dev/null 2>&1 || ip link show dev "$PEER_VETH" >/dev/null 2>&1 || ip link show dev "$HOST_IFB" >/dev/null 2>&1; then
        echo "An experiment interface name already exists; inspect before setup." >&2
        return 1
    fi
    trap cleanup ERR
    ip netns add "$NET_NS"
    ip link add "$HOST_VETH" type veth peer name "$PEER_VETH"
    ip link set "$PEER_VETH" netns "$NET_NS"
    ip addr add "$HOST_IP/30" dev "$HOST_VETH"
    ip link set "$HOST_VETH" mtu 1500 up
    ip netns exec "$NET_NS" ip addr add "$PEER_IP/30" dev "$PEER_VETH"
    ip netns exec "$NET_NS" ip link set "$PEER_VETH" mtu 1500 up
    ip netns exec "$NET_NS" ip link set lo up
    ip link add "$HOST_IFB" type ifb
    ip link set "$HOST_IFB" up
    ip netns exec "$NET_NS" ip link add "$PEER_IFB" type ifb
    ip netns exec "$NET_NS" ip link set "$PEER_IFB" up
    # Disable packet coalescing so the emulated link sees MTU-sized packets.
    ethtool -K "$HOST_VETH" tso off gso off gro off
    ip netns exec "$NET_NS" ethtool -K "$PEER_VETH" tso off gso off gro off
    tc qdisc add dev "$HOST_VETH" handle ffff: ingress
    tc filter add dev "$HOST_VETH" parent ffff: protocol all pref 1 matchall \
        action mirred egress redirect dev "$HOST_IFB"
    ip netns exec "$NET_NS" tc qdisc add dev "$PEER_VETH" handle ffff: ingress
    ip netns exec "$NET_NS" tc filter add dev "$PEER_VETH" parent ffff: protocol all pref 1 matchall \
        action mirred egress redirect dev "$PEER_IFB"
    profile "$1"
    trap - ERR
    echo "Created $NET_NS: host=$HOST_IP, server2=$PEER_IP"
}

case ${1:-} in
    setup) setup "${2:-}" ;;
    profile) profile "${2:-}" ;;
    status)
        ip netns list
        tc -s qdisc show dev "$HOST_IFB"
        ip netns exec "$NET_NS" tc -s qdisc show dev "$PEER_IFB"
        ;;
    cleanup) cleanup ;;
    *) usage; exit 1 ;;
esac

if [[ $link == client && ( ${1:-} == setup || ${1:-} == profile ) ]]; then
    ip netns exec armor-client ip route replace 10.203.77.1/32 via 10.203.78.1
fi
