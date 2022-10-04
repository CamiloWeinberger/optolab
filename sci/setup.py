from setuptools import find_packages, setup

setup(
  name='sci',
  packages=find_packages(),
  version='0.1.0',
  description='sci: a python package for ',
  author='Matias Valdivia',
  license='GNU Lesser General Public License v3.0',
  install_requires=[],
)

import os
env = f'LD_LIBRARY_PATH={os.getenv("LD_LIBRARY_PATH") + ":" if os.getenv("LD_LIBRARY_PATH") is not None else ""}{os.environ["CONDA_PREFIX"]}/lib/'
with open('.env', 'w') as f:
  f.write(env)

run_output = f'''conda activate {os.getenv("CONDA_PREFIX")}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
'''
with open('run_activate.sh', 'w') as f:
  f.write(run_output)