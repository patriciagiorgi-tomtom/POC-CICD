import numpy as np
import pyspark.sql
import pyspark.sql.functions as F

from pyspark.sql.types import StructType, StructField, DoubleType, BinaryType


import comlink
import asyncio
import equalaser
import acircuit


from equalaser._calibration import AzureLidarCalibrationDownloader, LidarCalibrationFetcherWithExternal


def get_transformation_matrixes(pc, cam_calib):
    pc_r = acircuit.mat_from_roll_pitch_heading(pc["roll"], pc["pitch"], pc["heading"] - np.pi / 2)
    pc_t = np.column_stack([pc["easting"], pc["northing"], pc["height"]]).astype(np.float32)

    # Projection matrices
    world_T_car = np.eye(4)
    world_T_car[0:3, 0:3] = pc_r
    world_T_car[0:3, -1] = pc_t
    car_T_world = np.linalg.inv(world_T_car)

    car_T_cam = np.eye(4)
    car_T_cam[0:3, -1] = cam_calib[0]
    car_T_cam[0:3, 0:3] = cam_calib[1]
    cam_T_car = np.linalg.inv(car_T_cam)

    # Combine
    cam_T_world = cam_T_car @ car_T_world

    return cam_T_world, car_T_world, pc_t


def process_session_trajectory(
    spark: pyspark.sql.SparkSession,
    session_name: str,
    carpositions_table_path: str,
    realignment_version: int,
    realignment_key: str,
    realignment_type: str,
    realignment_tsUrl: str,
    azure_lidar_calibration_sas: str,
):
    carpositions = (
        spark.read.format("delta").load(carpositions_table_path).select("session_name", "frame_index", "frametime")
    )
    carpositions = carpositions.filter(F.col("session_name") == session_name).drop_duplicates()

    session = comlink.SessionName(session_name)
    track_source = comlink.TrackSource(
        version=int(realignment_version), key=realignment_key, type_=realignment_type, url=realignment_tsUrl
    )
    track_source = track_source
    calibration_fetcher = LidarCalibrationFetcherWithExternal(
        ext_calibration_downloader=AzureLidarCalibrationDownloader(
            azure_lidar_calibration_sas,
        )
    )
    sess_info = asyncio.run(
        equalaser._utils.SessInfo.fetch(session, calibration_fetcher=calibration_fetcher, track_source=track_source)
    )
    trace_track = sess_info.track
    trace_track_broadcast = spark.sparkContext.broadcast(trace_track)
    cam_calib = asyncio.run(comlink._panoramix.fetch_ladybug_calibration(session))
    cam_calib_broadcast = spark.sparkContext.broadcast(cam_calib)
    origin_lon, origin_lat = trace_track.geo_to_world.lonlat_origin

    track_info_schema = StructType(
        [
            StructField("distance_along", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("height", DoubleType(), True),
            StructField("roll", DoubleType(), True),
            StructField("pitch", DoubleType(), True),
            StructField("heading", DoubleType(), True),
            StructField("easting", DoubleType(), True),
            StructField("northing", DoubleType(), True),
            StructField("cam_T_world", BinaryType(), True),
            StructField("car_T_world", BinaryType(), True),
            StructField("pc_t", BinaryType(), True),
        ]
    )

    @F.udf(track_info_schema)
    def process_track(time_in_seconds):
        trace_info = trace_track_broadcast.value[time_in_seconds]
        cam_calib = cam_calib_broadcast.value
        cam_T_world, car_T_world, pc_t = get_transformation_matrixes(trace_info, cam_calib)
        return (
            trace_info["distance_along"].item(),
            trace_info["longitude"].item(),
            trace_info["latitude"].item(),
            trace_info["height"].item(),
            trace_info["roll"].item(),
            trace_info["pitch"].item(),
            trace_info["heading"].item(),
            trace_info["easting"].item(),
            trace_info["northing"].item(),
            cam_T_world.tobytes(),
            car_T_world.tobytes(),
            pc_t.tobytes(),
        )

    carpositions = (
        carpositions.withColumn("origin_latitude", F.lit(origin_lat))
        .withColumn("origin_longitude", F.lit(origin_lon))
        .withColumn("time_only", F.date_format("frametime", "HH:mm:ss.SSS"))
        .withColumn(
            "frame_time_in_seconds",
            F.split(F.col("time_only"), ":")[0].cast("int") * 3600  # hours to seconds
            + F.split(F.col("time_only"), ":")[1].cast("int") * 60  # minutes to seconds
            + F.split(F.split(F.col("time_only"), ":")[2], "\.")[0].cast("int")  # seconds
            + F.split(F.split(F.col("time_only"), ":")[2], "\.")[1].cast("double") / 1000,
        )
    )
    carpositions_track = carpositions.withColumn("trace_attributes", process_track(F.col("frame_time_in_seconds")))
    carpositions_track = carpositions_track.withColumn("gps_antenna_height", F.lit(sess_info.gps_antenna_height))
    carpositions_track = carpositions_track.select(
        "session_name",
        "frame_index",
        "frame_time_in_seconds",
        "origin_latitude",
        "origin_longitude",
        "gps_antenna_height",
        F.col("trace_attributes.distance_along").alias("distance_along"),
        F.col("trace_attributes.longitude").alias("longitude"),
        F.col("trace_attributes.latitude").alias("latitude"),
        F.col("trace_attributes.height").alias("height"),
        F.col("trace_attributes.roll").alias("roll"),
        F.col("trace_attributes.pitch").alias("pitch"),
        F.col("trace_attributes.heading").alias("heading"),
        F.col("trace_attributes.easting").alias("easting"),
        F.col("trace_attributes.northing").alias("northing"),
        F.col("trace_attributes.cam_T_world").alias("cam_T_world"),
        F.col("trace_attributes.car_T_world").alias("car_T_world"),
        F.col("trace_attributes.pc_t").alias("pc_t"),
    )
    return carpositions_track
