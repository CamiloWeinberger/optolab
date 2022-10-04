# check if conda environment pyramidal is active and deactivate it
if [ "$CONDA_DEFAULT_ENV" = "pyramidal" ]; then
    echo "Deactivating conda environment pyramidal"
    conda deactivate
fi
# check if conda environment pyramidal exists and remove it
if conda env list | grep -q "pyramidal"; then
    echo "Removing conda environment pyramidal"
    conda env remove -n pyramidal
fi
# check if conda environment model_tonet is active and deactivate it
if [ "$CONDA_DEFAULT_ENV" = "model_tonet" ]; then
    echo "Deactivating conda environment model_tonet"
    conda deactivate
fi
# check if conda environment model_tonet exists and remove it
if conda env list | grep -q "model_tonet"; then
    echo "Removing conda environment model_tonet"
    conda env remove -n model_tonet
fi
# check if mlflow.service is active and stop it
if systemctl is-active --quiet mlflow.service; then
    echo "Stopping mlflow.service"
    sudo systemctl stop mlflow.service
fi
# check if mlflow.service exists and remove it
if systemctl list-unit-files | grep -q mlflow.service; then
    echo "Removing mlflow.service"
    sudo systemctl disable mlflow.service
    sudo rm /etc/systemd/system/mlflow.service
fi
# check if expose_mlflow.service is active and stop it
if systemctl is-active --quiet expose_mlflow.service; then
    echo "Stopping expose_mlflow.service"
    sudo systemctl stop expose_mlflow.service
fi
# check if expose_mlflow.service exists and remove it
if systemctl list-unit-files | grep -q expose_mlflow.service; then
    echo "Removing expose_mlflow.service"
    sudo systemctl disable expose_mlflow.service
    sudo rm /etc/systemd/system/expose_mlflow.service
fi