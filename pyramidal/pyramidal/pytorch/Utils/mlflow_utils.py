import os
from os.path import basename, join
from glob import glob
import json

def get_tracking_uri():
  # sentences for use local address or ngrok address
  is_ngrok = glob(join(os.path.expanduser('~'), '.mlflow', '.model_tonet', '*.use_internet_address'))[0]
  is_ngrok = int(basename(is_ngrok).split('.')[0])
  if is_ngrok:
    # load data.json from /home/$USER/.mlflow/.model_tonet/data.json
    with open(join(os.path.expanduser('~'), '.mlflow', '.model_tonet', 'data.json'), 'r') as f:
      ip_address = json.load(f)
    # load current user name for link train
    user_link = glob('*.server')[0].split('.')[0]
    return ip_address[user_link]
  else:
    with open(glob('*.server')[0], 'r') as f:
      user_link = f.read().strip()
    return f'http://{user_link}:5005'