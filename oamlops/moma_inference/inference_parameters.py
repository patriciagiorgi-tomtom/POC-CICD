import ast
from dataclasses import dataclass


@dataclass
class InferenceJobParameters:
    inference_data_output_path: str
    mlflow_experiment_name: str
    confidence_threshold: str
    model_mlflow_run_id: str
    inference_data_input_path: str
    mlflow_run_name: str
    petastorm_cache_dir_path: str
    inferencer: str
    device: str
    session_name: str
    run_id: str
    databricks_job_run_id: str
    databricks_job_parent_run_id: str
    run_metadata_table_path: str
    panorama_zoom_level: str
    panorama_zoom_size: tuple
    original_image_size: tuple = (5120, 2560)  # (width, height)
    rescale_image_size: tuple = (2666, 1333)
    inference_batch_size: int = 8
    model_batch_size: int = 8

    def __post_init__(self):
        if self.rescale_image_size[0] == 2666:
            self.inference_batch_size = 9
            self.model_batch_size = 9
        elif self.rescale_image_size[0] == 2000:
            self.inference_batch_size = 16
            self.model_batch_size = 16


class JobParametersProcessor:

    def __init__(self, job_parameters: InferenceJobParameters):
        self.job_parameters = job_parameters

    def get_checkpoint_run_id(
        self, task_key: str = "train_mmdetection_model", parameter_key: str = "mlflow_run_id"
    ) -> str:
        # Setting mlflow run id based on the previous notebooks
        try:
            run_id = dbutils.jobs.taskValues.get(taskKey=task_key, key=parameter_key)
        except:
            run_id = self.job_parameters.model_mlflow_run_id

        assert run_id is not None, "mlflow_run_id should not be None"
        assert len(run_id) > 0, "mlflow_run_id should not be empty"

        return run_id

    def set_ci_cd_pipeline(
        self, task_key: str = "train_mmdetection_model", parameter_key: str = "ci_cd_pipeline"
    ) -> str:
        try:
            ci_cd_pipeline = ast.literal_eval(dbutils.jobs.taskValues.get(taskKey=task_key, key=parameter_key))
        except:
            ci_cd_pipeline = ast.literal_eval(self.job_parameters.ci_cd_pipeline)
        return ci_cd_pipeline
