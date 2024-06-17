import requests
import pandas as pd
import h3
import shapely.geometry
import shapely.wkt
import json
import os

import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql

@F.udf(T.ArrayType(T.StringType()))
def get_moma_sessions_for_wkt(poligon_wkt: str) -> list[str]:
    url = "http://moma-api.sso.maps.az.tt3.com/api/search/FindSessions"
    params = {
        "key": "w3r3B1qzAW3IwSgCzPRq", #TODO: add in secrets
        "radius": "0",
        "wkt": poligon_wkt,
        "fromDate": "2021-01-01T00:00:00",
        "toDate": "2099-12-31T23:59:59"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.json_normalize(response.json())
        return df['SessionName'].tolist()
    else:
        []

def get_h3_indexes(polygon):
    lon_lat_coords = list(polygon.exterior.coords)
    lat_lon_coords = [(lat, lon) for lon, lat in lon_lat_coords]
    region_geojson = shapely.geometry.mapping(shapely.geometry.Polygon(lat_lon_coords))
    h3_indexes = h3.polyfill(region_geojson, 7)
    return list(h3_indexes)

@F.udf(T.StringType())
def get_h3_polygon(h3_index):
    hex_boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    polygon = shapely.geometry.Polygon(hex_boundary)
    return polygon.wkt

@F.udf(T.StringType())
def get_sqlite_path(session_name):
    url = f"https://panoserve.sso.maps.az.tt3.com/panoserve/{session_name}/directurl"
    params = {
        "filetype": "sqlite"
    }
    response = requests.get(url, params=params)
    if response.status_code==200:
        url = response.json()['url']
        return url
    else:
        return None
    
def extract_sessions_from_polygon(spark:pyspark.sql.SparkSession, polygon_wkt: str) -> list[str]:
    h3_indexes = get_h3_indexes(shapely.wkt.loads(polygon_wkt))
    print(f"Total h3 indexes to request {len(h3_indexes)}")

    schema = T.StructType([
        T.StructField("h3_index", T.StringType(), True)
    ])
    regions_df = spark.createDataFrame(pd.DataFrame({"h3_index":h3_indexes}), schema).repartition(len(h3_indexes)//8)
    regions_df = regions_df.withColumn("region_wkt", get_h3_polygon("h3_index"))
    regions_df = regions_df.withColumn("list_moma_sessions", get_moma_sessions_for_wkt("region_wkt"))
    regions_df.cache()
    moma_unique = regions_df.select(F.explode('list_moma_sessions').alias('moma_session')).distinct()
    total_moma_session = moma_unique.count()
    print(f"Total moma sessions for the region {total_moma_session}")
    
    return [x['moma_session'] for x in moma_unique.collect()]



def get_realigned_sessions(session_names):
    realigned_sessions = []
    url = "https://s2s-results-index-prod.mfs-master.sti-prod.tthad.net/realignment/last"
    for i in range(0, len(session_names), 1000):
        chunk = session_names[i:i+1000]  # Get a chunk of 1000 session_names (max allowed by the endpoint)
        
        realignments = requests.post(url, json=chunk)
        if realignments.status_code != 200:
            print(f"Error gettin realigment: {realignments.content}")
            return pd.DataFrame()
        realigned_sessions.append(pd.json_normalize(realignments.json()))
    
    realigned_sessions =  pd.concat(realigned_sessions)
    realigned_sessions = realigned_sessions[(realigned_sessions["key"]=="trajectory")&(realigned_sessions["type"]=="s2s")]
    return realigned_sessions

def copy_sqlite(session_name, destination_folder, destination_folder_sas_token):
    for extension in ['SQLITE', 'LSQLITE']:
        response = requests.get(f"https://panoserve.sso.maps.az.tt3.com/panoserve/{session_name}/directurl?filetype={extension}")
        url = json.loads(response.content)['url']
        status = os.system(f'/tmp/moma_raw/azcopy copy --overwrite=false "{url}" "{destination_folder.replace("abfss://", "https://")}?{destination_folder_sas_token}" &')
        if status != 0:
            print(f"There was an error when copying the files for session_name={session_name}, destination_folder={destination_folder} and destination_folder_sas_token={destination_folder_sas_token}.")
            return status
    return 0

def get_workflows_to_run(metadata, 
                     session_name, 
                     realignment_version, 
                     panorama_zoom_level, 
                     inference_mlflow_model_run_id):

    run_trajectory = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="trajectory")
        .filter(F.col("realignment.version")==realignment_version)
        .count() == 0
    ) 

    run_panorama = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="panorama")
        .filter(F.col("panorama.zoom_level")==panorama_zoom_level)
        .count()  == 0
    )

    run_lidar_polars = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="lidar_polars")
        .count()  == 0
    )

    run_inference = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="inference")
        .filter(F.col("panorama.zoom_level") == panorama_zoom_level)
        .filter(F.col("inference.mlflow_model_run_id") == inference_mlflow_model_run_id)
        .count() == 0
    ) 

    run_lidar = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="lidar_xyz")
        .filter(F.col("realignment.version") == realignment_version)
        .filter(F.col("inference.mlflow_model_run_id") == inference_mlflow_model_run_id)
        .count() == 0
    )
    
    run_carpositions = (
        metadata
        .filter(F.col("session_name")==session_name)
        .filter(F.col("event")=="done")
        .filter(F.col("workflow")=="carpositions")
        .count() == 0
    )

    return run_carpositions, run_trajectory, run_panorama, run_inference, run_lidar_polars, run_lidar
