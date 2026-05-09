# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME
echo "SENTINEL_FILE: $SENTINEL_FILE"

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if [[ ! -f $CONDA_ROOT/bin/mamba ]]; then
      $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    fi
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.11 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  pip install pyperclip
  # Then install isaacsim from NVIDIA index only
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  if [[ ! -d $WORKSPACE_DIR/IsaacLab ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 $WORKSPACE_DIR/IsaacLab
  fi

  # Only install cmake/build-essential via sudo if missing, so the script can
  # run unattended on machines that already have them.
  if ! command -v cmake >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
    sudo apt install -y cmake build-essential
  fi
  cd $WORKSPACE_DIR/IsaacLab
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  # isaaclab.sh calls `tabs 4` which fails under TERM=dumb, and triggers the
  # Omniverse EULA prompt on first launch. Set both up-front so it can run
  # non-interactively.
  export TERM=${TERM:-xterm-256color}
  export OMNI_KIT_ACCEPT_EULA=1
  export PRIVACY_CONSENT=Y
  # IsaacLab's pyproject pins flatdict==4.0.1 which is sdist-only on PyPI; pip's
  # build-isolation env grabs setuptools>=82 which removed pkg_resources, so the
  # build fails. Pin setuptools<82 in the build env via PIP_CONSTRAINT.
  CONSTRAINT_FILE=$(mktemp)
  echo "setuptools<82" > "$CONSTRAINT_FILE"
  PIP_CONSTRAINT="$CONSTRAINT_FILE" ./isaaclab.sh --install
  # The find...exec loop inside isaaclab.sh silently swallows individual install
  # failures, so the core `isaaclab` package is sometimes skipped. Install it
  # explicitly so `import isaaclab` works.
  PIP_CONSTRAINT="$CONSTRAINT_FILE" pip install -e $WORKSPACE_DIR/IsaacLab/source/isaaclab
  rm -f "$CONSTRAINT_FILE"

  # Install Holosoma
  pip install -U 'pip' 'setuptools<82'
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Force upgrade wandb to override rl-games constraint
  pip install --upgrade 'wandb>=0.21.1'
  # Holosoma pins numpy==1.23.5; isaaclab's install step bumps it to 1.26.x, so
  # restore it here. Pin setuptools<82 too because `pkg_resources` is still
  # imported by some pip-installed packages at runtime (e.g. flatdict 4.0.1).
  pip install 'numpy==1.23.5' 'setuptools<82'
  touch $SENTINEL_FILE
fi
