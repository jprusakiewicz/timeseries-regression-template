import sys

import mlflow
from omegaconf import OmegaConf

sys.path.append('src')

from data.preprocess import Preprocessor
from data.read_data import read_data
from data.data_models import PredData


def run_prediction(pred_data: PredData, model_id: str, model_library: str):
    mlflow.set_tracking_uri("http://localhost:50000")

    model_uri = f'runs:/{model_id}/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Predict on a Pandas DataFrame.
    if model_library == "sklearn":
        result = loaded_model.predict(pred_data.x_train_2d_p)
    elif model_library == "keras":
        result = loaded_model.predict(pred_data.xtensor_train_p)[0]
    else:
        raise ValueError
    return result


def run_predict(model_id: str):
    config_path = f"mlruns/0/{model_id}/artifacts/config.json"

    try:
        config = OmegaConf.load(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model with id: {model_id} not found")

    data = read_data()
    predicted_results = []
    for i in range(50):
        from datetime import datetime, timedelta

        last_date_str = data.iloc[len(data.index)-1, 0]
        date_format = "%Y-%m-%d"
        date_obj = datetime.strptime(last_date_str, date_format)

        # Add one day to get the next date
        next_date_obj = date_obj + timedelta(days=1)

        # Convert the datetime object back to a string
        next_date_str = next_date_obj.strftime(date_format)

        # df.loc[len(df.index)] = ['Amy', 89, 93]
        preprocessor = Preprocessor(config.preprocessing)
        pred_data = preprocessor.preprocess_predict_data(data)
        result = run_prediction(pred_data, model_id, config.model.model_library)
        data.loc[len(data.index)] = [next_date_str, result[0]]
        predicted_results.extend(result)
    return predicted_results


if __name__ == "__main__":
    model_id = "a6ecd31c064b41e985f0d93d7d91cc39"
    predictions = run_predict(model_id)
    print(predictions)
