#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case ${1:---help} in
    setup|profile) action=$1; shift; exec bash "$script_dir/network.sh" "$action" lan "$@" ;;
    status|cleanup) exec bash "$script_dir/network.sh" "$@" ;;
    keycenter|server2|server1|client) exec bash "$script_dir/role.sh" lan "$@" ;;
    *) echo 'Usage: bash lan.sh setup|profile|status|cleanup|keycenter|server2|server1|client [--dry-run] [Python options]' ;;
esac
