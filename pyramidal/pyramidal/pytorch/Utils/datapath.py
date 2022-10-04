# from pathlib import Path

import os
_user = os.environ['USER']
datapath = f'/home/{_user}/database/.camilo/Datasets_phase2pyr_May_2022_500modes'

assert datapath != None, 'datapath must be set'