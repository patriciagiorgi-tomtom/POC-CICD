import pyspark.sql.functions as F
import pyspark
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
)


def create_prediction_df(
    spark: pyspark.sql.session.SparkSession,
    image_detections: list,
    frame_indices: list,
    job_params: InferenceJobParameters,
    rescale_bbox_by_zoom_level: bool = True,
) -> pyspark.sql.DataFrame:
    """
    Function to create a spark dataframe from the inference results.

    Parameters:
    - spark (pyspark.sql.session.SparkSession): Spark session being used
    - image_detections (list): List of dictionaries containing the detection results
    - frame_indices (list): List of dictionaries containing the frame indices
    - job_params (InferenceJobParameters): Dataclass containing the job parameters

    Returns:
    - pyspark.sql.DataFrame: Spark dataframe containing the inference results

    """

    # Create flattened lists from batch data
    flattened_predictions = [
        prediction for batch_predictions in image_detections for prediction in batch_predictions["predictions"]
    ]
    flattened_indices = [
        {"frame_index": frame_index} for batch_frames in frame_indices for frame_index in batch_frames
    ]

    # Create a list of dictionaries with all the data
    prediction_data = list(zip(flattened_predictions, flattened_indices))
    prediction_dict = [{**detections, **indices} for detections, indices in prediction_data]

    # Create raw spark dataframe from dict
    predictions_df = spark.createDataFrame(prediction_dict)
    predictions_df = predictions_df.withColumn("join_index", F.monotonically_increasing_id())

    score_df = predictions_df.select("join_index", F.posexplode("scores").alias("pos", "score"))
    label_df = predictions_df.select("join_index", F.posexplode("labels").alias("pos", "class"))
    bbox_df = predictions_df.select("join_index", F.posexplode("bboxes").alias("pos", "bbox"))

    pred_joined_df = score_df.join(label_df, on=["join_index", "pos"]).join(bbox_df, on=["join_index", "pos"])
    predictions_df = pred_joined_df.join(
        predictions_df.select("frame_index", "join_index"),
        on=["join_index"],
    )
    predictions_df = predictions_df.filter(F.col("score") > job_params.confidence_threshold).drop("pos", "join_index")

    # Add a new column "class" based on the "id" column
    # predictions_df = predictions_df.withColumn("class", F.when(F.col("id") == 0, "traffic-sign").otherwise("unknown"))

    # BBOX: [x1, y1, x2, y2] | bbox_x = (x1 + x2) / 2 | bbox_y = (y1 + y2) / 2 | bbox_w = x2 - x1 | bbox_h = y2 - y1

    rescale_width_ratio = job_params.original_image_size[0] / job_params.rescale_image_size[0]
    rescale_height_ratio = job_params.original_image_size[1] / job_params.rescale_image_size[1]
    if rescale_bbox_by_zoom_level:
        rescale_width_ratio *= job_params.panorama_zoom_size[0] / job_params.original_image_size[0]
        rescale_height_ratio *= job_params.panorama_zoom_size[1] / job_params.original_image_size[1]

    predictions_df = (
        predictions_df.withColumn("bbox_x", (F.col("bbox")[0] + F.col("bbox")[2]) * rescale_width_ratio / 2)
        .withColumn("bbox_y", (F.col("bbox")[1] + F.col("bbox")[3]) * rescale_height_ratio / 2)
        .withColumn("bbox_w", (F.col("bbox")[2] - F.col("bbox")[0]) * rescale_width_ratio)
        .withColumn("bbox_h", (F.col("bbox")[3] - F.col("bbox")[1]) * rescale_height_ratio)
        .drop("bbox")
    )

    return predictions_df
