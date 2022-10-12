path = '/home/camilow/Desktop/CamiloW/2022/Test_27092022'

import scipy.io as sio
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from IPython.display import display, clear_output

# open file .amt as function
def load_mat(filename):
      data = sio.loadmat(filename)
      return data['X_phase'], data['X_s'], data['Y_z'], data['Y_p']

# names .mat files in order ascendent
file = os.listdir(path)[0]
print(file)
if fnmatch.fnmatch(file, '*.mat'):
    print(os.path.join(path, file))
    (X_phase, X_s, Y_z, Y_p) = load_mat(os.path.join(path, file))
    print(X_phase.shape)
    X_phase = X_phase.reshape(X_phase.shape[-1],X_phase.shape[0], X_phase.shape[1])
    X_s = X_s.reshape(X_s.shape[-1],X_s.shape[0], X_s.shape[1])


logged_model = 'runs:/8a9d7183781845968c753ffd5f53b497/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# data to predict
data = X_s[1,:,:]

# Predict on a Pandas DataFrame.
loaded_model.predict(pd.DataFrame(data))

