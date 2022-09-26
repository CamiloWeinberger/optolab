# check if conda environmet is installed
if [ -z "$CONDA_PREFIX" ]; then
    echo "Conda environment is not installed. Please install conda environment first."
    exit 1
fi
# install environment or update if already installed
if [ -d "$CONDA_PREFIX/envs/pytorch-env" ]; then
    echo "Updating pytorch-env environment"
    conda env update -n pytorch-env -f environment.yml
else
    echo "Installing pytorch-env environment"
    conda env create -n pytorch-env -f environment.yml
fi
# check if pyramidal package python is installed
if [ -z "$PYRAMIDAL_PYTHON" ]; then
    echo "Pyramidal package python is not installed. Please install pyramidal package python first."
    # install it
    pip install -e .
fi

