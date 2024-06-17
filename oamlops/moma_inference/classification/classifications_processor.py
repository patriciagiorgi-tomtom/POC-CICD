import pyspark.sql.functions as F
import pyspark
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
)


def create_classification_df(
        spark: pyspark.sql.session.SparkSession,
        image_predictions: list,
        indices: list,
        classes_dict: dict,
) -> pyspark.sql.DataFrame:
    flattened_predictions = [
        {
            "class_label": prediction["pred_label"],
            "class_score": prediction["pred_score"],
        }
        for batch_predictions in image_predictions
        for prediction in batch_predictions
    ]

    flattened_indices = [
        {"bbox_index": bbox_index}
        for batch_indices in indices
        for bbox_index in batch_indices
    ]
    prediction_data = list(zip(flattened_predictions, flattened_indices))
    prediction_dict = [{**classes, **indices} for classes, indices in prediction_data]
    classification_df = spark.createDataFrame(prediction_dict)
    classification_df = classification_df.withColumn(
        "class_label", F.udf(lambda x: classes_dict.get(x, "unknown"), "string")(F.col("class_label").cast("string"))
    )

    return classification_df
