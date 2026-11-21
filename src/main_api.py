from threading import Lock
import mlflow
from pyspark.sql import SparkSession
from flask import Flask, request, jsonify
from http import HTTPStatus

from configs.configs import run_config
from src.data_preprocessing import get_nested_column_extended
from src.feature_extractor import FeatureExtractor
from src.mlflow_utils import get_latest_run_id
from src.model import Model


def get_model(run_id=None) -> Model:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    if run_id is None:
        run_id = get_latest_run_id(run_config.run_name)

    model = Model()
    model.load_model_from_mlflow(run_id)
    return model


def get_feature_extractor_loaded(run_id=None) -> FeatureExtractor:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    if run_id is None:
        run_id = get_latest_run_id(run_config.run_name)

    feature_extractor = FeatureExtractor()
    feature_extractor.load_from_mlflow(run_id)
    return feature_extractor


# no cache so we control which object serves request
_state_lock = Lock()
_reload_lock = Lock()

mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
SparkSession.builder.master(run_config.spark_master_url).getOrCreate()

initial_run_id = get_latest_run_id(run_config.run_name)
_model = get_model(initial_run_id)
_feature_extractor = get_feature_extractor_loaded(initial_run_id)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return "OK", HTTPStatus.OK


@app.route("/predict", methods=["POST"])
def predict():
    with _state_lock:
        model = _model
        feature_extractor = _feature_extractor

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


@app.route("/reload", methods=["POST"])
def reload_model():
    global _model, _feature_extractor

    with _reload_lock:
        try:
            run_id = get_latest_run_id(run_config.run_name)

            # new model and extractoer objects (while predictions still use the old ones)
            new_model = get_model(run_id)
            new_feature_extractor = get_feature_extractor_loaded(run_id)

        except Exception:
            app.logger.exception("Model reload failed")
            return jsonify(
                message="new model reload failed - keeping current model"
            ), HTTPStatus.SERVICE_UNAVAILABLE

        with _state_lock:
            _model = new_model
            _feature_extractor = new_feature_extractor

    return jsonify(message="Model reloaded", run_id=run_id), HTTPStatus.OK
