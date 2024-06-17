# Clone of https://github.com/tomtom-internal/uhd_map/blob/main/road_furniture/mmdet_comp/visualization.py
# This bypasses the error coming from mmdet in mlflow

import os
import os.path as osp

import numpy as np

from mmengine import VISBACKENDS, MMLogger
from mmengine.visualization import MLflowVisBackend
from mmengine.visualization.vis_backend import force_init_env
from mmengine.utils import scandir


@VISBACKENDS.register_module()
class MLflowVisBackendDatabricks(MLflowVisBackend):
    """
    We are overwriting this as the original class tries to set self._mlflow.set_tracking_uri(self._tracking_uri) which
    leads to major inconsistencies on Databricks. So we removed this from `_init_env`.
    """

    def _init_env(self):
        """Setup env for MLflow."""

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow'
            )  # type: ignore
        self._mlflow = mlflow

        if "mlflow_run_name" in os.environ:
            self._run_name = os.environ["mlflow_run_name"]

        if "mlflow_experiment_name" in os.environ:
            self._exp_name = os.environ["mlflow_experiment_name"]

        # when mlflow is imported, a default logger is created.
        # at this time, the default logger's stream is None
        # so the stream is reopened only when the stream is None
        # or the stream is closed
        logger = MMLogger.get_current_instance()
        for handler in logger.handlers:
            if handler.stream is None or handler.stream.closed:
                handler.stream = open(handler.baseFilename, 'a')

        self._exp_name = self._exp_name or 'Default'

        if "mlflow_run_id" in os.environ:
            mlflow_run_id = os.environ["mlflow_run_id"]
            self._mlflow.start_run(run_id=mlflow_run_id)

        else:
            if self._mlflow.get_experiment_by_name(self._exp_name) is None:
                self._mlflow.create_experiment(self._exp_name)

            self._mlflow.set_experiment(self._exp_name)

        if self._run_name is not None:
            self._mlflow.set_tag('mlflow.runName', self._run_name)
        if self._tags is not None:
            self._mlflow.set_tags(self._tags)
        if self._params is not None:
            self._mlflow.log_params(self._params)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to mlflow.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Default to 0.
        """
        self._mlflow.log_image(image, f"validation_images/{step}/{name}")

    def close(self) -> None:
        """Close the mlflow."""
        if not hasattr(self, '_mlflow'):
            return

        # work_dir/{model_name}/inference/{timestamp}/vis_data
        timestamp = self._save_dir.split("/")[-2]
        base_folder = osp.join(self.cfg.work_dir, timestamp)

        file_paths = dict()
        for filename in scandir(base_folder, self._artifact_suffix, True):
            file_path = osp.join(base_folder, filename)
            relative_path = os.path.relpath(file_path, base_folder)
            dir_path = os.path.dirname(relative_path)
            file_paths[file_path] = dir_path

        print(file_paths)
        for file_path, dir_path in file_paths.items():
            print(f"Logging {file_path} to {dir_path}")
            print(f"File exists: {file_path}", os.path.isfile(file_path))

            try:
                self._mlflow.log_artifact(file_path, dir_path)
            except Exception as e:
                print(f"Failed to log {file_path} to {dir_path} with Error: {e}")

        # self._mlflow.end_run()
