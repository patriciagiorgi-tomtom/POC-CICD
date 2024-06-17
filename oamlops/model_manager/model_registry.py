from dataclasses import dataclass, asdict
import mlflow
from mmdet.apis import DetInferencer
import os
from oamlops.model_manager.mlflow_model import MMDetectionModel
from oamlops.model_manager.model_classes.image_classification_inferencer import ImageClassificationInferencer
from oamlops.model_manager.model_classes.tensor_det_inferencer import TensorDetInferencer
from oamlops.model_manager.model_utils import get_classifier_classes_from_config, get_path_to_the_final_checkpoint, get_config_as_dict

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
    model_config: dict | None = None
    classes: dict | None = None


    def __post_init__(self):
        self.inference_image_width = int(self.inference_image_width)
        self.inference_image_height = int(self.inference_image_height)
        self.checkpoint_artifact_dir_path: str = os.path.join(self.model_temp_dir_path, self.checkpoint_mlflow_run_id)
        self.model_log_artifact_path = rf"{self.model_log_artifact_path}/{self.inferencer_type}"
        if self.model_name is None:
            self.model_name = self.inferencer_type
        try:
            self.classes = get_classifier_classes_from_config(self.model_configs_path)
        except Exception as e:
            print(f"Error in getting class dictionary : {e}")
        self.model_config = get_config_as_dict(self.model_configs_path)
        # # TODO: Use config to update the width and height of inference image
        # self.inference_image_width = self.model_config['val_dataloader']['dataset']['pipeline'][1]['scale'][0]
        # self.inference_image_height = self.model_config['val_dataloader']['dataset']['pipeline'][1]['scale'][1]


class ModelRegistry:
    inferencer_dict = {"DetInferencer": DetInferencer, 
                       'TensorDetInferencer' : TensorDetInferencer, 
                       'ImageClassificationInferencer': ImageClassificationInferencer }

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
    ) :
        inferencer_class = self.inferencer_dict[self.parameters.inferencer_type]

        if self.parameters.inferencer_type == 'ImageClassificationInferencer':
            inferencer = inferencer_class(
                        model=self.parameters.model_configs_path,
                        pretrained=get_path_to_the_final_checkpoint(self.parameters.checkpoint_artifact_dir_path),
                        device="cpu" if "cpu" in self.parameters.device_type.lower() else f"cuda:{device_rank}",
                        )
            return inferencer
        
        inferencer = inferencer_class(
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
