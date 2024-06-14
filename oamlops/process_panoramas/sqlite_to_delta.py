import argparse
import os
import io
import numpy as np
import re
import math
from data_loader.save_panorama import PanoramaProviderLocal
from data_loader.sqlite_data_provider import SqliteDataProvider
from moma_sqlite_data_handlers.handlers.panorama import Panorama
from moma_sqlite_data_handlers.handlers.camera import Camera

import pyspark.sql
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, TimestampType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import BinaryType
from PIL import Image
import sys
import logging


def set_storage_account_config(
    spark,
    storage_account: str,
    client_id: str,
    tenant_id: str,
    service_credential,
):
    """Configures the spark session for a datalake connection

    Args:
        spark (pyspark.sql.session.SparkSession): Spark session being used
        storage_account (str): The storage account name being configured
        client_id (str): Client id for the datalake connection
        tenant_id (str): Tenant id for the datalake connection
        service_credential (_type_): The service credential created for the connection

    Returns:
        pyspark.sql.session.SparkSession: Configured spark session
    """
    logging.warning("DEPRECATED. Use databricks_common.py instead")
    # perform authentication
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
    spark.conf.set(
        f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
        "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net",
        client_id,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net",
        service_credential,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net",
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
    )


def get_date_from_session_name(session_name):

    pattern = r"(\d{4})_(\d{2})_(\d{2})"  # Extract date
    match = re.search(pattern, session_name)
    if match:
        date_from_session = tuple(map(int, match.groups()))
    date_from_session_str = f"{date_from_session[0]}-{date_from_session[1]:02d}-{date_from_session[2]:02d}"
    return date_from_session_str


def get_session_name_from_path(session_path):
    session_name = session_path.split("/")[-1].split(".")[0]
    return session_name


def retrieve_car_positions_for_sessions(spark: pyspark.sql.SparkSession, moma_session_paths: list[str]):
    carpositions_df = None

    for session_path in moma_session_paths:
        session_name = get_session_name_from_path(session_path)
        date_from_session_str = get_date_from_session_name(session_name)
        df = (
            spark.read.format("jdbc")
            .option("url", f"jdbc:sqlite:{session_path}")
            .option("dbtable", "carpositions")
            .load()
            .withColumn("frametime", F.concat(F.lit(date_from_session_str), F.lit(" "), "frametime"))
            .withColumn("frametime", F.to_timestamp("frametime", "yyyy-MM-dd HH:mm:ss.SSS"))
            .withColumn("session_name", F.lit(session_name))
            .withColumn("idx", F.col("idx").cast("integer"))
            .withColumnRenamed("idx", "frame_index")
        )

        if carpositions_df:
            carpositions_df = carpositions_df.union(df)
        else:
            carpositions_df = df

    return carpositions_df


def retrieve_camera_frames(
    spark: pyspark.sql.SparkSession, moma_session_paths: list[str], batch_size: int, zoom_level: int
):
    cameraframes_df = None

    for session_path in moma_session_paths:

        session_name = get_session_name_from_path(session_path)
        ppl = PanoramaProviderLocal(session_path, zoom_level, "panorama")
        camera_info = ppl.sqlite_data_provider.get_camera_by_type("panorama")

        df = (
            spark.read.format("jdbc")
            .option("url", f"jdbc:sqlite:{session_path}")
            .option("dbtable", f"cameraframes{zoom_level}")
            .load()
            .withColumnRenamed("idx", "frame_index")
            .withColumn("session_name", F.lit(session_name))
            .withColumn("camera", F.lit(camera_info))
            .withColumn("zoom_level", F.lit(zoom_level))
            .withColumn("compression", F.lit(ppl.compression))
        )

        if cameraframes_df:
            cameraframes_df = cameraframes_df.union(df)
        else:
            cameraframes_df = df

    total_frames = cameraframes_df.count()
    cameraframes_df = cameraframes_df.repartition(total_frames // batch_size, "frame_index")
    return cameraframes_df


def get_image_as_stream(
    panorama: np.ndarray, rescale_size: tuple, target_format="JPEG", quality=95, resize: bool = False
):
    """Get assembled panorama as bytes.

    :param str target_format:
    :param int quality: JPEG quality parameter
    (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving).
    :return: Panorama data as bytes.
    :returns: bytes
    """
    img = Image.fromarray(panorama)
    if resize:
        img = img.resize(size=rescale_size)
    img_data = io.BytesIO()
    img.save(img_data, format=target_format, quality=quality)
    return img_data.getvalue()


def get_panorama_img_udf(rescale_size: tuple, resize: bool = False):
    @F.udf("binary")
    def get_panorama_img(group):
        sorted_group = sorted(group, key=lambda x: x.part)
        sorted_ttvalues = [item.ttvalue for item in sorted_group]
        camera_info = sorted_group[0].camera
        zoom_level = sorted_group[0].zoom_level
        panorama = Panorama(Camera(camera_info), zoom_level, sorted_ttvalues).image_data
        return get_image_as_stream(panorama=panorama, rescale_size=rescale_size, resize=resize)

    return get_panorama_img


def process_frames(
    cameraframes_df: pyspark.sql.DataFrame, rescale_size: tuple, resize: bool = False
) -> pyspark.sql.DataFrame:
    get_panorama_img = get_panorama_img_udf(rescale_size, resize)
    frames_df = (
        cameraframes_df.groupby("session_name", "frame_index", "zoom_level", "compression")
        .agg(F.collect_list(F.expr("struct(part, ttvalue, camera, zoom_level)")).alias("group"))
        .withColumn("panorama_img", get_panorama_img("group"))
        .drop("group")
    )
    return frames_df
