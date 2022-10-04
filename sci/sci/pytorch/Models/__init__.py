import os
import sys
import importlib
from torch import nn

def load_model(model_name) -> nn.Module:
  r"""
  Load the model.

  Args:
    model_name: Name of the model.

  Returns:
    The model.

  Raises:
    ValueError: If the model is not implemented.
  """
  model_path = f'{os.path.join(os.path.dirname(__file__), model_name)}.py'
  if not os.path.exists(model_path):
    raise ValueError(f'{model_name} is not implemented')

  spec = importlib.util.spec_from_file_location(model_name, model_path)
  model_container = importlib.util.module_from_spec(spec)
  sys.modules[model_name] = model_container
  spec.loader.exec_module(model_container)
  model_to_use = model_container.__dict__[model_name]
  return model_to_use