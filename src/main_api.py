from functools import cache

import mlflow
from pyspark.sql import SparkSession
from flask import Flask, request, jsonify
from http import HTTPStatus

from configs.configs import run_config
from src.data_preprocessing import get_nested_column_extended
from src.feature_extractor import FeatureExtractor
from src.mlflow_utils import get_latest_run_id
from src.model import Model


@cache
def get_model() -> Model:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    model = Model()
    model.load_model_from_mlflow(run_id)
    return model


@cache
def get_feature_extractor_loaded() -> FeatureExtractor:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    feature_extractor = FeatureExtractor()
    feature_extractor.load_from_mlflow(run_id)
    return feature_extractor


# Doing "starting up tasks" like loading the model and feature extractor to cache before the app is started.
# So that the first request wont have a big latency spike.
get_model()
get_feature_extractor_loaded()
SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return "OK", HTTPStatus.OK


@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    feature_extractor = get_feature_extractor_loaded()
    spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
    request_data = spark.createDataFrame(
        [
            {
                "amount": request.json["amount"],
                "category": request.json["category"],
                "payment_type": request.json["payment_type"],
            },
        ]
    )
    features = feature_extractor.get_features(request_data)
    features = features.toPandas()
    for col_name in ["enc_category", "enc_payment_type"]:
        features = get_nested_column_extended(features, col_name)
    prediction = model.predict(features)
    return jsonify(prediction[0].item())
