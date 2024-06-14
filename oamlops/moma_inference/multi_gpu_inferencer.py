import itertools
import torch
from oamlops.road_furniture.mmdet_comp.det_inferencer import DetInferencer
from oamlops.road_furniture.tools.tensor_det_inferencer import TensorDetInferencer
from oamlops.road_furniture.utils.general_utils import get_path_to_the_final_checkpoint
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
)
from oamlops.moma_inference.moma_inferencer import batch_inference
from oamlops.moma_inference.petastorm_processor import (
    get_tens_transform_spec,
    get_transform_spec,
)
from petastorm.spark import SparkDatasetConverter


def moma_inference_multi_gpu(
    job_parameters: InferenceJobParameters,
    model_configs: str,
    checkpoint_artifact_dir_path: str,
    converter_inference_data: SparkDatasetConverter,
) -> tuple:
    """
    This function is used to perform inference on multiple GPUs using Horovod

    Example usage: >>

    hr = HorovodRunner(np=cluster_avail_gpu)
    inference_detections, inference_frame_indices, image_sizes, session_indices, final_execution_time = hr.run(
        main=moma_inference_multi_gpu, job_parameters=job_params,
            notebook_dir_path=notebook_path,
            checkpoint_artifact_dir_path=checkpoint_artifact_dir_path,
            converter_inference_data=converter_inference_data
    )

    """
    import horovod.torch as hvd
    import os
    import time
    from sparkdl.horovod import log_to_driver

    hvd.init()
    rank = hvd.rank()
    world_size = hvd.size()
    if rank == 0:
        log_to_driver(f"Cluster size: {world_size} GPUs")

    # set device
    device = "cpu" if "cpu" in job_parameters.device.lower() else f"cuda:{hvd.local_rank()}"
    if "cuda" in device:
        torch.cuda.set_device(hvd.local_rank())
    torch.cuda.empty_cache()

    moma_inferencer = TensorDetInferencer if job_parameters.inferencer == "TensorDetInferencer" else DetInferencer

    inferencer = moma_inferencer(
        model=model_configs,
        weights=get_path_to_the_final_checkpoint(checkpoint_artifact_dir_path),
        show_progress=False,
        device=device,
    )
    transform_func = (
        get_tens_transform_spec() if job_parameters.inferencer == "TensorDetInferencer" else get_transform_spec()
    )
    with converter_inference_data.make_torch_dataloader(
        cur_shard=rank,
        shard_count=world_size,
        transform_spec=transform_func,
        batch_size=int(job_parameters.inference_batch_size),
        num_epochs=1,
    ) as inference_data_loader:
        start_time = time.time()
        inference_detections, inference_frame_indices = batch_inference(
            inference_data_loader, inferencer, int(job_parameters.model_batch_size)
        )

        end_time = time.time()
        log_to_driver("\nWorker %i: Duration: %f" % (rank, round(end_time - start_time, 3)))

        # Gather data from all GPUs
        all_detections = hvd.allgather_object(inference_detections)
        all_frame_indices = hvd.allgather_object(inference_frame_indices)

        final_detections = list(itertools.chain.from_iterable(all_detections))
        final_frame_indices = list(itertools.chain.from_iterable(all_frame_indices))

    return final_detections, final_frame_indices
