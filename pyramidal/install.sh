# check if conda is installed
if ! which conda > /dev/null; then
    echo "conda is not installed! please install anaconda first"
    exit 1
fi
# check if environment pyramidal is already installed if not is installed install it with environment.yml file
if ! conda env list | grep -q "^pyramidal "; then
    echo "Installinng pyramidal environment..."
    conda env create -f env.yml
fi
# check if environment is activated if not activate it
if ! conda info --envs | grep -q "^\* pyramidal "; then
    echo "Activating pyramidal environment..."
    conda activate pyramidal
fi