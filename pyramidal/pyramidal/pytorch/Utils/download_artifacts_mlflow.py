import os

def compile_download_artifacts_mlflow(mlflow_tracking_uri, project, run_id, datavariant, mlflow):
  template_to_download_artifacts = f"""import mlflow
  mlflow_tracking_uri = '{mlflow_tracking_uri}'
  project = '{project}'
  run_id = '{run_id}'
  datavariant = '{datavariant}'

  while True:
    print('Folder must be one of: all, details, model, parameters, results, by default is results')
    folder = input('Folder to download artifacts [results]: ')
    if folder in ['details', 'model', 'parameters', 'results', '', 'all']:
      if folder == '':
        folder = 'results'
      elif folder == 'all':
        folder = ''
      print(f'Downloading artifacts from {{mlflow_tracking_uri}} to /tmp/mlflow/outputs/{{project}} {{datavariant}}/{{folder}}')
      break

  path_downloaded = mlflow.artifacts.download_artifacts(f'runs:/{{run_id}}/{{folder}}', dst_path=f'/tmp/mlflow/outputs/{{project}} {{datavariant}}/')

  print(f'Artifacts downloaded into {{path_downloaded}}')

  r = input('Do you want to move the downloaded artifacts to the Desktop folder? [y/N]: ')

  if r.strip() in ['y', 'Y']:
    import os
    import shutil
    is_desktop_or_escritorio = lambda x: x in ['Desktop', 'Escritorio']
    desktop = list(filter(is_desktop_or_escritorio, os.listdir(f'/home/{{os.getenv(\"USER\")}}/')))
    if len(desktop) == 0:
      print('Desktop folder not found')
    else:
      desktop = desktop[0]
      print(f'Moving downloaded artifacts to /home/{{os.getenv(\"USER\")}}/{{desktop}}/{{project}}/{{datavariant}}')
      os.makedirs(f'/home/{{os.getenv(\"USER\")}}/{{desktop}}/{{project}}/{{datavariant}}', exist_ok=True)
      shutil.move(path_downloaded, f'/home/{{os.getenv(\"USER\")}}/{{desktop}}/{{project}}/{{datavariant}}')
      print('Done')

  # Outputs:
  # Folder must be one of: all, details, model, parameters, results, by default is results
  # Folder to download artifacts [results]:
  # Downloading artifacts from http://localhost:5005 to /tmp/mlflow/outputs/pyramidal 54/results
  # Artifacts downloaded into /tmp/mlflow/outputs/pyramidal 54/results
  # Do you want to move the downloaded artifacts to the Desktop folder? [y/N]:
  # Moving downloaded artifacts to /home/USER/Escritorio/pyramidal/54
  # Done

  """.replace('\n  ', '\n')

  os.makedirs('/tmp/mlflow/save', exist_ok=True)

  with open('/tmp/mlflow/save/download_artifacts.py', 'w') as f:
    f.write(template_to_download_artifacts)

  mlflow.log_artifact('/tmp/mlflow/save/download_artifacts.py', 'download_artifacts')

if __name__ == '__main__' and False:
  import mlflow
  mlflow_tracking_uri = 'http://localhost:5005'
  compile_download_artifacts_mlflow('http://localhost:5005', 'pyramidal', '57ca7c05039f44f3b87ebd4f2adffa5c', '54', mlflow)

def download_all_artifacts(mlflow_tracking_uri, project='pyramidal'):
  import os
  import mlflow

  while True:
    print('Folder must be one of: [all, details, model, parameters, results] by default is results')
    folder = input('Folder to download artifacts [results]: ')
    if folder in ['details', 'model', 'parameters', 'results', '', 'all']:
      if folder == '':
        folder = 'results'
      elif folder == 'all':
        folder = ''
      print(f'Downloading artifacts from {mlflow_tracking_uri} to current folder')
      break

  mlflow.set_tracking_uri(mlflow_tracking_uri)
  all_experiments = [exp.experiment_id for exp in mlflow.list_experiments()]
  all_experiments.pop(0)
  for run in mlflow.search_runs(experiment_ids=all_experiments).iloc:
    if run['tags.mlflow.source.name'].split('/')[0] == project:
      if run['status'] == 'FINISHED':
        modelName = run['tags.modelName']
        loss_fn = run['params.loss_fn']
        datavariant = run['params.datavariant']
        model_name = ''
        multiply_filters = ''
        if run['params.model_name'] != None:
          model_name = f"_{run['params.model_name']}"
          multiply_filters = f"_{run['params.multiply_filters']}"

        dst_path = f'./{modelName}_{loss_fn}_{datavariant}{model_name}{multiply_filters}'
        os.makedirs(dst_path, exist_ok=True)

        run_id = run['run_id']
        path_downloaded = mlflow.artifacts.download_artifacts(f'runs:/{run_id}/{folder}', dst_path=dst_path)

        print(f'Artifacts downloaded into {path_downloaded}')

if __name__ == '__main__':
  host = 'https://ff26-158-251-246-24.sa.ngrok.io'
  download_all_artifacts(host)