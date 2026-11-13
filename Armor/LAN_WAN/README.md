# ARMOR: LAN and WAN

Use this directory for encrypted ARMOR with four processes on one Linux GPU host.
Use `../main.py` for single-process training. Both modes share `../train.py`,
`../Attacks`, `../dataset.py`, and `../Ciphertext`; no attack code is copied here.

## Files

| File | Purpose |
| --- | --- |
| `lan.sh` / `wan.sh` | Select LAN or WAN and run setup, a role, or cleanup |
| `role.sh` | Common launcher for all four roles |
| `network.sh` | Configure both emulated network links |
| `config.env` | Shared endpoints, GPUs, training, and attack defaults |
| `keycenter.py` | Generate parameters and distribute key material |
| `server1.py` | Filter and aggregate encrypted updates |
| `server2.py` | Perform partial decryption |
| `client.py` | Train logical clients sequentially and encrypt their updates |
| `distributed.py` / `_shared.py` | Transport helpers and shared-code paths |

## 1. Prepare

Follow the [repository installation instructions](../../README.md#installation) first.
Build `Armor/phantom-fhe` on the target machine after downloading the repository. Activate the Python
environment matching `Armor/phantom-fhe/build/lib/pyPhantom*.so`.
All four roles need the CUDA runtime and the matching `libPhantom.so` build.

From the repository root:

```bash
cd Armor/LAN_WAN
```

Network emulation requires Linux, sudo, iproute2 (`ip` and `tc`), ethtool,
iputils-ping, and kernel support for network namespaces, veth, IFB, and netem.
On systems where IFB is not loaded, run `sudo modprobe ifb` first.

## 2. Set up one profile

Choose **one**:

```bash
bash lan.sh setup
# OR:
bash wan.sh setup
```

Both profiles shape Client–Server1 and Server1–Server2 using isolated virtual
links. They leave the physical network interface unchanged.

| Profile | Rate per direction, per link | Added delay per direction | Added RTT per link |
| --- | --- | --- | --- |
| LAN | 3000 Mbit/s | 0.1 ms | 0.2 ms |
| WAN | 100 Mbit/s | 40 ms | 80 ms |

These are single-host emulation settings, not measured goodput or a physical
multi-machine deployment. Server2 runs in namespace `armor-s2`; Client runs in
`armor-client`. KeyCenter and Server1 run in the host namespace.

Check the settings with `bash lan.sh status` (or `bash wan.sh status`). If setup
already exists, use `bash lan.sh profile` or `bash wan.sh profile` to change it,
after stopping all roles.

## 3. Start four terminals

Generate one authentication key:

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

In **each terminal**, activate the same environment, enter `Armor/LAN_WAN`, and
set the same key and a fresh run tag:

```bash
export ARMOR_AUTHKEY='paste-the-same-generated-key-in-all-four-terminals'
export RUN_TAG=lan_mnist_rep1
export ARMOR_PYTHON="$(command -v python)"
```

Start the roles in this order, one command per terminal:

```bash
# Terminal 1
bash lan.sh keycenter
# Terminal 2
bash lan.sh server2
# Terminal 3
bash lan.sh server1
# Terminal 4
bash lan.sh client
```

For WAN, use `wan.sh` in all four commands after setting up the WAN profile,
and choose a corresponding fresh tag such as `wan_mnist_rep1`.
A role may wait for another role to start; this is expected.
Server2 and Client request sudo to enter their network namespaces.

Defaults: MNIST, 3 logical clients, 2 global rounds, local epoch 1, batch size 32,
seed 2023, GPU 0 for every role, and no attack. A single Client process executes
all logical clients. Edit `config.env` to change the common defaults, or use
Python options on the KeyCenter command:

```bash
bash lan.sh keycenter --dataset MNIST --num_clients 10 --global_ep 3 \
  --attack MPAF --attack_ratio 0.3 --attack_time 0
```

Keep the other three role commands unchanged: KeyCenter distributes experiment
settings. `--attack_time` is zero based; 0 starts in the first round.
Other attacks and their parameters are described in the repository README.
Use FMNIST or CIFAR10 only with sufficient GPU memory.

To select GPUs, set the same configuration in each terminal, for example:

```bash
export KEYCENTER_GPU=0 SERVER1_GPU=1 SERVER2_GPU=2 CLIENT_GPU=3
```

## 4. Check commands without running

```bash
bash lan.sh keycenter --dry-run
bash lan.sh client --dry-run
bash wan.sh server2 --dry-run
```

Dry runs print commands without GPU work, sockets, sudo, or network changes.
They do not check whether namespaces exist or whether the profile is applied.
`lan.sh` and `wan.sh` select launch defaults; use `setup` or `profile` explicitly
to apply network shaping.

## 5. Results and cleanup

Each role prints progress to its terminal. Runtime events are generated in
`logs/${RUN_TAG}_{role}.jsonl` under this directory. The launcher passes a
literal `{role}` template, and each process substitutes its own role name.
Choose a fresh `RUN_TAG` for each run.

Historical experiment results are also included in `ARMOR_FMNIST_clients_logs/`,
`ARMOR_Lancelot_logs/`, `ARMOR_WAN_BY_DATASET/`, `BreakDown/`, and `Ablation/`.
Retained training accuracy results are in `../accuracy_logs/`. These historical
results are not inputs required to launch the four roles.

Client and Server1 may download datasets into the repository's `data/` directory
on first use. For an offline run, populate that cache beforehand. All four roles
share the GPU when using the defaults, so leave enough memory for the combined
processes.

Stop all roles with Ctrl+C in their terminals before changing network settings.
Then remove both virtual links:

```bash
bash lan.sh cleanup
```

`wan.sh cleanup` performs the same cleanup. If a run fails, check the failing
terminal first, then confirm the authentication key, interpreter, GPU IDs,
endpoints, and network setup match across all four terminals.
