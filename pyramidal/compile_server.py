import os

user = os.getenv('USER')
with open(f'{user}.server', 'w') as f:
  f.write('')