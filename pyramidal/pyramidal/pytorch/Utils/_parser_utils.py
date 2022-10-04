

def get_useful_kwargs(kwargs, file_name):
  with open(file_name, 'r') as f:
    lines = f.readlines()
  # search add_argument and add argument name to kwargs with value from kwargs
  usefull_kwargs = {}
  for l in lines:
    if '.add_argument(' in l and l.strip().startswith('#') == False:
      # get argument after "--"
      # for example: "    parser.add_argument('--hidden_dim', type=int, default=128)" will be "hidden_dim"
      arg_name = l.split('--')[1].split(',')[0].strip().replace('"', '').replace("'", '').replace('-', '_')
      usefull_kwargs[arg_name] = kwargs[arg_name]
  base_args = ['lr', 'normalize_head', 'normalize_tail', 'datavariant', 'is_float']
  for arg in base_args:
    usefull_kwargs[arg] = kwargs[arg]
  return usefull_kwargs

if __name__ == '__main__':
  from os.path import join, dirname
  kwargs = {'hidden_dim': 128, 'lr': 0.001, 'batch_size': 64, 'epochs': 10, 'seed': 42}
  get_useful_kwargs(kwargs, join(dirname(dirname(__file__)), 'Models/mnist_model.py'))