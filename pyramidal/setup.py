from setuptools import find_packages, setup

setup(
  name='pyramidal',
  packages=find_packages(),
  version='0.1.0',
  description='pyramidal: a python package for ',
  author='Matias Valdivia',
  license='GNU Lesser General Public License v3.0',
  install_requires=[],
)

import os
from os.path import join
from glob import glob
user_server = glob('*.server')[0].split('.')[0]
if user_server.find(' ') != -1:
  user_server = user_server.split(' ')[0]
conda_path = os.environ["CONDA_PREFIX"]

# installing ngrok for remote access to mlflow server from local machine (only for user server)
# install environment "env_model_tonet.yml" with conda
# check if model_tonet environment is installed
expose = False
user = os.environ["USER"]
os.makedirs(f'/home/{user}/.mlflow/.model_tonet/', exist_ok=True)
if os.path.exists('expose.py'):
  expose = True
  if not os.path.exists(f'{os.getenv("CONDA_PREFIX_1")}/envs/model_tonet'):
    os.system('conda env create -f env_model_tonet.yml')
  path_python_model_tonet_env = join(os.getenv('CONDA_PREFIX_1'), 'envs/model_tonet/bin')
  # mkdir for expose.py
  # move expose.py to .mlflow/.model_tonet
  os.system(f'mv expose.py /home/{user_server}/.mlflow/.model_tonet/')
  # move worker.py to .mlflow/.model_tonet
  os.system(f'mv worker.py /home/{user_server}/.mlflow/.model_tonet/')
  # move optolab-pucv.json to .mlflow/.model_tonet
  os.system(f'mv optolab-pucv.json /home/{user_server}/.mlflow/.model_tonet/')
  # move env_model_tonet.yml to .mlflow/.model_tonet
  os.system(f'mv env_model_tonet.yml /home/{user_server}/.mlflow/.model_tonet/')

if user_server == os.getenv('USER'):
  os.system(f'hostname -I > {glob("*.server")[0]}')
  with open(glob("*.server")[0], 'r') as f:
    lines = f.readlines()

  # sometimes you will have two o more IP addresses, so this app will take the first one
  for l in lines:
    if len(l.strip().split(' ')) > 1:
      l = l.split(' ')[0]
  with open(glob("*.server")[0], 'w') as f:
    f.write(l)

  if os.path.exists(f'/etc/systemd/system/mlflow.service'):
    os.system('sudo systemctl stop mlflow.service')
    os.system('sudo systemctl disable mlflow.service')
    os.system('sudo rm /etc/systemd/system/mlflow.service')

  service = f'''Description=MLflow Tracking Server
  After=network.target

  [Service]
  Restart=on-failure
  RestartSec=30
  User={user_server}
  Group={user_server}
  Type=simple
  StandardOutput=syslog
  ExecStart=/bin/bash -c 'PATH={conda_path}/bin/:$PATH exec mlflow server --backend-store-uri sqlite:///home/{user_server}/.mlflow/metrics.db --artifacts-destination /home/{user_server}/.mlflow/artifacts --serve-artifacts -h 0.0.0.0 -p 5005'

  [Install]
  WantedBy=multi-user.target'''.replace('\n  ', '\n')
  with open('mlflow.service', 'w') as f:
    f.write(service)
  os.system('sudo mv mlflow.service /etc/systemd/system/')
  os.system('sudo systemctl daemon-reload')
  os.system('sudo systemctl enable mlflow')
  os.system('sudo systemctl start mlflow')

  if expose:

    if os.path.exists(f'/etc/systemd/system/expose_mlflow.service'):
      os.system('sudo systemctl stop expose_mlflow.service')
      os.system('sudo systemctl disable expose_mlflow.service')
      os.system('sudo rm /etc/systemd/system/expose_mlflow.service')

    service = f'''Description=pyNgrok expose mlflow server and upload public url to firebase
    After=network.target

    [Service]
    Restart=on-failure
    RestartSec=30
    User={user_server}
    Group={user_server}
    Type=simple
    StandardOutput=syslog
    ExecStart=/bin/bash -c 'PATH={path_python_model_tonet_env}:$PATH exec python /home/{user_server}/.mlflow/.model_tonet/expose.py'

    [Install]
    WantedBy=multi-user.target'''.replace('\n  ', '\n')
    with open('expose_mlflow.service', 'w') as f:
      f.write(service)
    os.system('sudo mv expose_mlflow.service /etc/systemd/system/')
    os.system('sudo systemctl daemon-reload')
    os.system('sudo systemctl enable expose_mlflow')
    os.system('sudo systemctl start expose_mlflow')

  # for server use local address to connect to mlflow server
  is_internet = 0
else:
  if expose:
    if os.path.exists(f'/etc/systemd/system/worker_mlflow.service'):
      os.system('sudo systemctl stop worker_mlflow.service')
      os.system('sudo systemctl disable worker_mlflow.service')
      os.system('sudo rm /etc/systemd/system/worker_mlflow.service')

    service = f'''Description=Get public url from firebase database for mlflow worker
    After=network.target

    [Service]
    Restart=on-failure
    RestartSec=30
    User={user_server}
    Group={user_server}
    Type=simple
    StandardOutput=syslog
    ExecStart={path_python_model_tonet_env}/python /home/{user_server}/.mlflow/.model_tonet/worker.py'

    [Install]
    WantedBy=multi-user.target'''.replace('\n  ', '\n')
    with open('worker_mlflow.service', 'w') as f:
      f.write(service)
    os.system('sudo mv worker_mlflow.service /etc/systemd/system/')
    os.system('sudo systemctl daemon-reload')
    os.system('sudo systemctl enable worker_mlflow')
    os.system('sudo systemctl start worker_mlflow')

  # use local ip or public ip
  is_internet = 0
with open(f'/home/{os.getenv("USER")}/.mlflow/.model_tonet/{is_internet}.use_internet_address', 'w') as f:
  f.write('')


with open(glob('*.server')[0]) as f:
  ip = f.readline().strip()

LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")


env = f'''LD_LIBRARY_PATH={LD_LIBRARY_PATH + ":" if LD_LIBRARY_PATH is not None else ""}{conda_path}/lib/
MLFLOW_TRACKING_URI=http://{ip}:5005
'''
with open('.env', 'w') as f:
  f.write(env)

if len(ip) == 0:
  raise Exception('Init server first!!')
run_output = f'''conda activate {conda_path}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export MLFLOW_TRACKING_URI=http://{ip}:5005
'''
with open('run_activate.sh', 'w') as f:
  f.write(run_output)