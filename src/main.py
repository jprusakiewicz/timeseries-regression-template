import glob
import json
import sys

from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline
import mlflow
from urllib3.exceptions import NewConnectionError
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('src')

from data.read_data import read_data
from data.preprocess import Preprocessor
from data.data_models import TrainingData
from models.sklearn import build_model as build_sklearn_model
from models.kerass import build_model as build_keras_model
from evaluate import evaluate_model


def run_experiment(training_data: TrainingData, config) -> dict:
    try:
        mlflow.set_tracking_uri("http://localhost:50000")
    except NewConnectionError as e:
        print("MLFLOW connection error. Is mlflow running?")
        raise e
    with mlflow.start_run(run_name=f"test_run"):
        config_as_json = OmegaConf.to_container(config)
        mlflow.log_dict(config_as_json, 'config.json')
        mlflow.set_tag('config_hash', hash(frozenset(config_as_json)))

        match config.model.model_library:
            case "keras":
                mlflow.tensorflow.autolog()
                model = build_keras_model(training_data.x_train_3d, training_data.y_train, config.model)
                x_test = training_data.x_test_3d

            case "sklearn":
                mlflow.sklearn.autolog()
                model = make_pipeline(build_sklearn_model(config.model))
                model.fit(training_data.x_train_2d, training_data.y_train)
                x_test = training_data.x_test_2d

            case _:
                raise ValueError(f"unsupported model library: {config.model.model_library}")

        metrics = evaluate_model(model=model, x=x_test, y=training_data.y_test)
        mlflow.log_metrics(metrics)

    return metrics


def get_json_files(directory):
    json_files = glob.glob(directory + '/*.json')
    return json_files


def run_from_yaml():
    metrics = run()
    print(metrics)


def run_from_jsons(parallel=False):
    json_files = get_json_files("config/generated_configs")
    merged_configs = []
    for file in json_files:
        with open(file, 'r') as json_file:
            data = json.load(json_file)
            merged_configs.extend(data)

    print(f"number of configs: {len(merged_configs)}")
    if parallel:
        metrics = Parallel(n_jobs=4)(
            delayed(run)(config) for config in tqdm(merged_configs)
        )
        print(metrics)
    else:
        for config in merged_configs:
            metrics = run(config)
            print(metrics)


def run(dict_config: dict | None = None):
    config_path = 'config/test_sklearn_config.yaml'

    if dict_config is None:
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create(dict_config)
    data = read_data()
    preprocessor = Preprocessor(config.preprocessing)

    training_data = preprocessor.preprocess_training_data(data)
    result = run_experiment(training_data, config)

    return result


if __name__ == "__main__":
    run_from_yaml()
    # run_from_jsons()
