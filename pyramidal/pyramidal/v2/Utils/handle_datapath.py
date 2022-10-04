import os

def handle_datapath(datapath=''):
  if len(datapath) > 0:
    db = f'_{datapath}'
  else:
    db = ''

  if os.path.expanduser('~').find('mvaldi') != -1:
    datapath = f'/home/mlflow/Datasets_phase2pyr_May_2022{db}_npy'
  elif os.path.expanduser('~').find('optolab') != -1:
    datapath = f'/home/optolab/Desktop/Camilo/Datasets_phase2pyr_May_2022{db}_npy'
  elif os.path.expanduser('~').find('bizon') != -1:
    datapath = f'/home/bizon/Documents/matias/Camilo/Datasets_phase2pyr_May_2022{db}_npy'
  else:
    raise Exception('Unknown user')

  return datapath