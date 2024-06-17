from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, LongType, BinaryType, StringType
import pandas as pd
from PIL import Image, ImageDraw
import io
import pyspark
import os
from oamlops.moma_inference.inference_parameters import InferenceJobParameters


def get_bbox_coordinates(x_center, y_center, w, h, rescale_width_ratio, rescale_height_ratio) -> tuple:
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return (
        xmin * rescale_width_ratio,
        ymin * rescale_height_ratio,
        xmax * rescale_width_ratio,
        ymax * rescale_height_ratio,
    )


def create_crop_images_udf(job_params: InferenceJobParameters):
    rescale_width_ratio = (
            job_params.rescale_image_size[0] / job_params.original_image_size[0]
    )
    rescale_height_ratio = (
            job_params.rescale_image_size[1] / job_params.original_image_size[1]
    )
    if job_params.rescale_bbox_by_zoom_level:
        rescale_width_ratio *= (
                job_params.original_image_size[0] / job_params.panorama_zoom_size[0]
        )
        rescale_height_ratio *= (
                job_params.original_image_size[1] / job_params.panorama_zoom_size[1]
        )

    @pandas_udf(ArrayType(StructType([
        StructField("bbox_index", LongType(), True),
        StructField("cropped_image", BinaryType(), True)
    ])))
    def crop_images(image_bytes: pd.Series, bboxes: pd.Series) -> pd.Series:
        output = []
        for i in range(len(image_bytes)):
            image = Image.open(io.BytesIO(image_bytes[i]))
            bbox_cropped_images = []
            for bbox in bboxes[i]:
                xmin, ymin, xmax, ymax = get_bbox_coordinates(
                    bbox['bbox_x'], bbox['bbox_y'], bbox['bbox_w'], bbox['bbox_h'], rescale_width_ratio,
                    rescale_height_ratio
                )
                cropped_img = image.crop((xmin, ymin, xmax, ymax))
                if job_params.save_bbox:
                    os.makedirs(os.path.dirname(bbox['bbox_temp_path']), exist_ok=True)
                    cropped_img.save(bbox['bbox_temp_path'])
                img_byte_arr = io.BytesIO()
                cropped_img.save(img_byte_arr, format="PNG")
                bbox_cropped_images.append((bbox['bbox_index'], img_byte_arr.getvalue()))
            output.append(bbox_cropped_images)
        return pd.Series(output)

    return crop_images


def process_bboxes(spark: pyspark.sql.session.SparkSession, job_parameters: InferenceJobParameters, predictions_df: pyspark.sql.DataFrame) -> (pyspark.sql.DataFrame, pyspark.sql.DataFrame) :

    selected_sessions = [job_parameters.session_name]
    moma_inference_df = (
        spark.read.format("delta")
        .load(job_parameters.inference_data_input_path)
        .filter(F.col("session_name").isin(selected_sessions))
        .filter(F.col("zoom_level") == job_parameters.panorama_zoom_level)
        .select("frame_index", "panorama_img")
        .repartition(2)
    )
    
    if job_parameters.save_bbox:
        predictions_df = predictions_df.withColumn(
                        "bbox_temp_path",
                        F.concat(
                            F.lit("/dbfs"),
                            F.lit(job_parameters.mount_destination_point), F.lit("/"),
                            F.col("session_name"), F.lit("/"),
                            F.col("mlflow_inference_run_id"), F.lit("/"),
                            F.col("bbox_index"), F.lit(".jpg")
                        )
                        ).withColumn("bbox_crop_path", 
                                     F.concat(
                                    F.lit(job_parameters.mount_source_point), F.lit("/"),
                                    F.col("session_name"), F.lit("/"),
                                    F.col("mlflow_inference_run_id"), F.lit("/"),
                                    F.col("bbox_index"), F.lit(".jpg")
                        ))
    else:
         predictions_df = predictions_df.withColumn(
                        "bbox_temp_path", F.lit(None)).withColumn(
                        "bbox_crop_path", F.lit(None))
    # Group the bounding boxes by frame_index
    bbox_df = predictions_df.groupBy('frame_index').agg(
        F.collect_list(F.struct("bbox_index", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_temp_path")).alias("bboxes"))

    # Join with moma_inference_df on frame_index
    bbox_df = bbox_df.join(moma_inference_df, on="frame_index")

    del moma_inference_df

    # Create the UDF
    crop_images_udf = create_crop_images_udf(job_parameters)

    # Apply the crop_image function
    bbox_df = bbox_df.withColumn("cropped_images", crop_images_udf(F.col("panorama_img"), F.col("bboxes")))

    # Drop the original image column
    bbox_df = bbox_df.drop("panorama_img", "bboxes")

    bbox_df = bbox_df.select(F.explode("cropped_images").alias("cropped_images")).select(
        "cropped_images.bbox_index",
        "cropped_images.cropped_image"
    )

    return bbox_df, predictions_df
