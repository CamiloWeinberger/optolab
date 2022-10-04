import os
from os.path import join

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository
from pytorch_lightning.loggers.mlflow import MLFlowLogger

class MLFlowModelCheckpoint(ModelCheckpoint):
  def __init__(self, mlflow_logger: MLFlowLogger, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mlflow_logger = mlflow_logger
    self._best_model_path_last = ''

    self.sftp_repository = None

  def init_sftp_repository(self):
    self.sftp_repository = SFTPArtifactRepository(join(os.getenv('MLFLOW_ARTIFACT_URI'), self.mlflow_logger._experiment_id, self.mlflow_logger.run_id, 'artifacts'))

  @rank_zero_only
  def on_validation_end(self, trainer, pl_module):
    super().on_validation_end(trainer, pl_module)
    if len(self.best_model_path) > 0:
      if self.best_model_path != self._best_model_path_last:
        if self.sftp_repository == None:
          self.init_sftp_repository()
        self._best_model_path_last = self.best_model_path
        self.sftp_repository.log_artifact(self.best_model_path)