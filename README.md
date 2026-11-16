# ARMOR: Toward Practical, Secure, and Robust Federated Learning against Model Poisoning Attacks via Homomorphic Encryption

## Overview

ARMOR is a privacy-preserving federated learning framework that defends against model poisoning attacks using homomorphic encryption.

## Installation

Run the commands below from the repository root (the directory containing
this README and `Armor/`). After downloading or cloning the repository, enter
that directory first. The native library must be rebuilt on the target machine;
local build outputs are not included.


### 1. Dependencies

#### System Requirements
- **CMake**: ≥ 3.20
- **Compiler**: GCC ≥ 9.4 (C++17 support required)
- **OS**: Linux (required for the LAN/WAN namespace scripts)
- **GPU**: NVIDIA GPU for encrypted execution
- **CUDA Toolkit**: `nvcc` and a compatible NVIDIA driver/host compiler
- **Python**: 3.12 (the environment used below)
- **Conda**: recommended(our setting)

#### GPU Architecture Mapping Table

Use the following table to find the correct value for `-DCMAKE_CUDA_ARCHITECTURES`.

| Representative GPU Models | Values (Architecture) |
| :--- | :---: |
| **NVIDIA H100, H200** | `90` |
| **RTX 6000 Ada, RTX 4090, L40, L4** | `89` |
| **NVIDIA A100**  | `80` |
| **RTX 3090, A6000** | `86` |
| **RTX 2080 Ti, T4, Quadro RTX 8000** | `75` |
| **NVIDIA V100**  | `70` |
| **GTX 1080 Ti, P100** | `61`, `60` |

### 2. Build ARMOR (phantom-fhe)
#### 2.1 Pre-build: CUDA Environment Check
Before proceeding, ensure that the CUDA compiler (`nvcc`) is correctly detected.
```bash
which nvcc
```
The path may differ depending on your CUDA installation. If `nvcc` is not found,
set the paths to your installed toolkit, for example:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

#### 2.2: Build phantom-fhe & pyPhantom 
```bash
conda create -n ARMOR python=3.12 -c conda-forge -y
conda activate ARMOR

cd Armor/phantom-fhe

export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
# CMake uses the bundled python/pybind11 source directory.

# Adjust for your GPU
cmake -S . -B build \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DPython3_ROOT_DIR="$CONDA_PREFIX" \
    -DPHANTOM_ENABLE_PYTHON_BINDING=ON

cmake --build build -j

SP_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$PWD/build/lib" > "$SP_DIR/phantom_fhe.pth"

# Return to the repository root for the commands below
cd ../..
```

### 3. Set Up Python Environment
```bash
conda install pip -y
python -m pip install numpy torch torchvision tensorboard

# Confirm that this environment can load the rebuilt binding and use the GPU.
python -c "import pyPhantom, torch; print(pyPhantom.__file__); print('CUDA available:', torch.cuda.is_available())"
```

Use a CUDA-enabled PyTorch build compatible with your environment. If the checkout
is moved, rebuild in the new location and update `phantom_fhe.pth` with the new
absolute build path. Datasets are downloaded automatically into `data/` at the
repository root on first use; internet access is needed unless they are already
cached there.

---
## Architecture
```text
ARMOR
├── README.md                  # Installation and experiment commands
│
└── Armor/
    ├── main.py                # Main entry point for system execution
    ├── options.py             # Hyperparameter and argument settings
    ├── supplementary_jobs.py  # Queue missing or incomplete supplementary experiments
    ├── armor_filter_config.py # Shared MAD and population STD thresholds
    ├── dataset.py             # Data loading and preprocessing
    ├── test.py                # Evaluation of global parameters and model performance
    ├── train.py               # Local training and shared attack execution
    ├── sampling.py            # Client data partitioning and sampling configuration
    ├── Attacks/               # Shared attack implementations used by main.py and LAN/WAN
    ├── phantom-fhe/           # Core CUDA-based library for Homomorphic Encryption (HE)
    │
    ├── Ciphertext/            
    │   ├── encrypt_decrypt/   
    │   │   ├── params.py      # HE parameter settings (Scale factor, Modulus chain, etc.)
    │   │   ├── encrypt.py     # Local update encryption via coefficient encoding
    │   │   └── decrypt.py     # Multi-party decryption and tensor restoration (Fitting)
    │   └── utils.py           # Norm check and similarity check in the ciphertext domain
    │
    ├── Plaintext/             
    │    ├── utils.py          # Defense methods (Krum, Multi-Krum, Cosine-defense, Vanilla ARMOR)
    │    └── intersection.py   # Plaintext ARMOR with intersected MAD and STD filters
    │
    ├── run_armor_cifar10.sh   # Shell script to run FedAvg and ARMOR protocol on CIFAR10 dataset
    ├── run_armor_mnist.sh     # Shell script to run FedAvg and ARMOR protocol on mnist dataset
    ├── run_armor_fmnist.sh    # Shell script to run FedAvg and ARMOR protocol on fmnist dataset
    │
    ├── LAN_WAN/               # Distributed roles, network profiles, and LAN/WAN README
    └── accuracy_logs/         # Saved accuracy results under poisonedFL, MPAF, scaling attacks.

```
---

## Running ARMOR
The ARMOR protocol's memory requirements vary depending on the model.
### MNIST(LeNet5) (<= 10GB VRAM)
```bash
bash Armor/run_armor_mnist.sh
```

### FMNIST(ResNet18) (40GB ~ 80GB VRAM)
```bash
bash Armor/run_armor_fmnist.sh
```

### CIFAR-10(ResNet34) (>= 80GB VRAM)
```bash
bash Armor/run_armor_cifar10.sh
```

---

The dataset launch scripts run PoisonedFL with `--poisonedfl_scale 8`.
ALIE remains available as an additional attack but is excluded from these scripts.

### Supplementary Experiments

Run `Armor/supplementary_jobs.py` from the repository root to execute missing or
incomplete supplementary experiments. The default queue contains 120 combinations:
three datasets (`MNIST`, `FMNIST`, `CIFAR10`), five attacks (`alie`,
`sign_flipping`, `label_flipping`, `minsum`, `minmax`), and eight methods (`armor`,
`vanilla_armor`, `fedavg`, `trimmed_mean`, `cosine`, `multi_krum`, `krum`, `median`).
Here, `armor` enables encrypted ARMOR; `vanilla_armor` runs plaintext ARMOR.

```bash
# Inspect pending experiments without starting training
python Armor/supplementary_jobs.py --dry-run

# Run the default queue on GPU 0
python Armor/supplementary_jobs.py --gpu 0

# Run a selected subset
python Armor/supplementary_jobs.py --datasets MNIST --attacks alie minmax \
    --methods vanilla_armor fedavg --gpu 0

# Rerun selected experiments, archiving existing logs
python Armor/supplementary_jobs.py --datasets MNIST --attacks alie \
    --methods vanilla_armor --rerun --gpu 0
```

Each experiment uses 50 clients, an attack ratio of 0.3, attack start time 50,
200 global rounds, one local epoch, batch size 32, and seed 2023. Accuracy logs
are saved under `Armor/accuracy_logs/SupplyMentary/<DATASET>/`. Full-length logs
are skipped by default; incomplete experiments restart from the beginning, with
previous logs moved to the dataset's `archive/` directory.

## Key Arguments

| Argument | Description |
|----------|------------|
| `--dataset` | `MNIST`, `FMNIST`, `CIFAR10` |
| `--attack_ratio` | Proportion of malicious clients among all clients |
| `--attack` | attack type (MPAF, scaling, poisonedfl; additional attacks supported) |
| `--method` | defense method(fedavg, armor) |
| `--armor_filter` | `default` (default) or `intersection`; the latter requires plaintext `--method armor` and combines MAD 3.5 and population STD 1.779 filters |
| `--cipher` | use homomorphic encryption |
---

## Main Attacks

- **PoisonedFL**: Malicious clients maintain a consistent attack direction across rounds to accumulate model drift, adapting the attack magnitude based on observed global model changes.
- **MPAF**: Malicious clients construct amplified updates using a randomly generated model to disrupt global model convergence.
- **Scaling**: Malicious clients amplify their local updates to dominate aggregation.

## Additional Attacks

- **ALIE**: Malicious clients use the mean and standard deviation of reference updates to craft small statistical deviations that evade robust aggregation.
- **Min-Sum**: Malicious clients shift updates away from the reference mean while constraining the sum of squared distances to reference updates.
- **Min-Max**: Malicious clients shift updates away from the reference mean while constraining the maximum squared distance to reference updates.
- **Sign flipping**: Malicious clients reverse the signs of their local updates to oppose the learning direction.
- **Label flipping**: Malicious clients train on deliberately incorrect labels to corrupt their local updates.

## Four-process Execution over LAN/WAN

See [LAN/WAN instructions](Armor/LAN_WAN/README.md) for the four-terminal setup
using `lan.sh` or `wan.sh`. Both modes share the training and attack code used
by `Armor/main.py`.

## Licences

ARMOR is distributed under the **GNU General Public License v3.0 (GPLv3)**.
See the [licence text](Armor/phantom-fhe/LICENSE).

Third-party code retains its original licences:

| Component | Licence |
| --- | --- |
| Modified PhantomFHE | [GPLv3](Armor/phantom-fhe/LICENSE) |
| Microsoft SEAL-derived code | MIT |
| OpenFHE-derived code | BSD 2-Clause |
| Bundled pybind11 | [BSD 3-Clause](Armor/phantom-fhe/python/pybind11/LICENSE) |

See the [PhantomFHE notices](Armor/phantom-fhe/README.md#license) for details.
Existing copyright, licence, and modification notices must be retained when redistributing the code.

## Attack References

- **PoisonedFL**: Yueqi Xie, Minghong Fang, and Neil Zhenqiang Gong. [Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Model_Poisoning_Attacks_to_Federated_Learning_via_Multi-Round_Consistency_CVPR_2025_paper.pdf). CVPR, 2025.
- **MPAF**: Xiaoyu Cao and Neil Zhenqiang Gong. [MPAF: Model Poisoning Attacks to Federated Learning Based on Fake Clients](https://openaccess.thecvf.com/content/CVPR2022W/FedVision/papers/Cao_MPAF_Model_Poisoning_Attacks_to_Federated_Learning_Based_on_Fake_CVPRW_2022_paper.pdf). CVPR Workshops, 2022.
- **Scaling (related work on update amplification)**: Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. [How To Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html). AISTATS, 2020. The scaling baseline here only amplifies local updates; it does not implement the paper's full backdoor attack.
- **ALIE**: Gilad Baruch, Moran Baruch, and Yoav Goldberg. [A Little Is Enough: Circumventing Defenses For Distributed Learning](https://papers.nips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html). NeurIPS, 2019.
- **Min-Sum / Min-Max**: Virat Shejwalkar and Amir Houmansadr. [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning](https://people.cs.umass.edu/~amir/papers/NDSS21-model-poisoning.pdf). NDSS, 2021.
- **Sign flipping / Label flipping (baseline descriptions, Appendix D.1)**: Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan. [Byzantine Machine Learning Made Easy By Resilient Averaging of Momentums](https://proceedings.mlr.press/v162/farhadkhani22a.html). ICML, 2022.
- **Label flipping (federated data poisoning)**: Vale Tolpegin, Stacey Truex, Mehmet Emre Gursoy, and Ling Liu. [Data Poisoning Attacks Against Federated Learning Systems](https://arxiv.org/abs/2007.08432). ESORICS, 2020.
