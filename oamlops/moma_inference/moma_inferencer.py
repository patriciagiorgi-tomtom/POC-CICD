import time
from mmdet.apis import DetInferencer
import torch
from petastorm.spark import SparkDatasetConverter
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
)
from oamlops.moma_inference.petastorm_processor import (
    get_transform_spec,
)
import mlflow


# Model does not support tensor currently, hence need to convert tensors back to numpy
def detect_signs(pd_batch, inferencer: DetInferencer, batch_size: int) -> list:
    detections = inferencer(list(pd_batch["image"].numpy()), show=False, batch_size=batch_size)
    return detections


def batch_inference(
    inference_data_loader: iter,
    inferencer: DetInferencer,
    model_batch_size: int,
) -> tuple:
    detections = []
    frame_indices = []

    for batch_id, pd_batch in enumerate(inference_data_loader):
        batch_detections = detect_signs(pd_batch, inferencer, model_batch_size)

        batch_frame_indices = pd_batch["frame_index"].squeeze().tolist()

        if pd_batch["frame_index"].numel() == 1:
            batch_frame_indices = [batch_frame_indices]
        # print(f"Finished processing {batch_id=}")
        detections.append(batch_detections)
        frame_indices.append(batch_frame_indices)
    return detections, frame_indices


def moma_inference_single_gpu(
    job_parameters: InferenceJobParameters,
    converter_inference_data: SparkDatasetConverter,
) -> tuple:
    device = "cpu" if "cpu" in job_parameters.device.lower() else f"cuda:{0}"

    if "cuda" in device:
        torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    model = mlflow.pyfunc.load_model(job_parameters.model_mlflow_run_id)
    inferencer = model.unwrap_python_model().inferencer
    with converter_inference_data.make_torch_dataloader(
        transform_spec=get_transform_spec(
            image_shape=job_parameters.rescale_image_size,
            resize=job_parameters.original_image_size != job_parameters.rescale_image_size,
        ),
        batch_size=int(job_parameters.inference_batch_size),
        num_epochs=1,
    ) as inference_data_loader:
        start_time = time.time()

        inference_detections, inference_frame_indices = batch_inference(
            inference_data_loader, inferencer, int(job_parameters.model_batch_size)
        )
        end_time = time.time()
        print(f"Duration: {round(end_time - start_time, 3)}")
    return inference_detections, inference_frame_indices


def execute_moma_inference(
    job_parameters: InferenceJobParameters,
    converter_inference_data: SparkDatasetConverter,
    model_configs: str | None = None,
    checkpoint_artifact_dir_path: str | None = None,
    number_of_gpu_workers: int = 1,
) -> tuple:
    """
    This function is used to perform inference on single or multiple GPUs using Horovod
    """

    if number_of_gpu_workers <= 0:
        raise ValueError("Number of GPU workers should be greater than 0")

    if number_of_gpu_workers > 1:
        from sparkdl import HorovodRunner
        from oamlops.moma_inference.multi_gpu_inferencer import (
            moma_inference_multi_gpu,
        )

        hr = HorovodRunner(np=number_of_gpu_workers)
        return hr.run(
            main=moma_inference_multi_gpu,
            job_parameters=job_parameters,
            model_configs=model_configs,
            checkpoint_artifact_dir_path=checkpoint_artifact_dir_path,
            converter_inference_data=converter_inference_data,
        )

    return moma_inference_single_gpu(
        job_parameters=job_parameters,
        converter_inference_data=converter_inference_data,
    )
