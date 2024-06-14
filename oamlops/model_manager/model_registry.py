from dataclasses import dataclass, asdict
import mlflow
from mmdet.apis import DetInferencer
import os
from oamlops.model_manager.mlflow_model import MMDetectionModel

def get_path_to_the_final_checkpoint(folder: str) -> str:
    # get all folders that start with model_
    model_folders = [x for x in os.listdir(folder) if x.startswith("model_")]
    # sort first by epoch and then by iteration
    model_folders.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
    # return the last folder
    return os.path.join(folder, model_folders[-1], "state_dict.pth")

@dataclass
class ModelRegistryParameters:
    inferencer_type: str
    checkpoint_mlflow_run_id: str
    model_temp_dir_path: str
    model_configs_path: str
    inference_image_width: int 
    inference_image_height: int
    device_type: str = "gpu"
    model_name: str | None = None
    model_log_artifact_path: str = "model"


    def __post_init__(self):
        self.inference_image_width = int(self.inference_image_width)
        self.inference_image_height = int(self.inference_image_height)
        self.checkpoint_artifact_dir_path: str = os.path.join(self.model_temp_dir_path, self.checkpoint_mlflow_run_id)
        self.model_log_artifact_path = rf"{self.model_log_artifact_path}/{self.inferencer_type}"
        if self.model_name is None:
            self.model_name = self.inferencer_type


class ModelRegistry:
    inferencer_dict = {"DetInferencer": DetInferencer}

    def __init__(self, parameters: ModelRegistryParameters):
        self.parameters = parameters

    def download_artifact(self) -> None:
        mlflow.artifacts.download_artifacts(
            run_id=self.parameters.checkpoint_mlflow_run_id,
            artifact_path="./",
            dst_path=self.parameters.checkpoint_artifact_dir_path,
        )

    def create_inferencer(
        self, device_rank: int = 0, show_progress: bool = False
    ) -> DetInferencer:
        inferencer = self.inferencer_dict[self.parameters.inferencer_type](
            model=self.parameters.model_configs_path,
            weights=get_path_to_the_final_checkpoint(self.parameters.checkpoint_artifact_dir_path),
            show_progress=show_progress,
            device="cpu" if "cpu" in self.parameters.device_type.lower() else f"cuda:{device_rank}",
        )
        return inferencer

    def register_model(self, download_artifact: bool = True) -> mlflow.models.model.ModelInfo:
        if download_artifact:
            self.download_artifact()
        inferencer = self.create_inferencer()
        mmdetection_model = MMDetectionModel(inferencer, self.parameters)
        model_info = mlflow.pyfunc.log_model(
            artifact_path=self.parameters.model_log_artifact_path,
            registered_model_name=self.parameters.model_name,
            python_model=mmdetection_model,
            metadata=asdict(self.parameters),
        )
        return model_info
