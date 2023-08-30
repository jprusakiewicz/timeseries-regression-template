import dataclasses
from typing import Any


@dataclasses.dataclass
class PredData:
    x_train_2d_p: Any
    xtensor_train_p: Any


@dataclasses.dataclass
class TrainingData:
    x_train_2d: Any
    x_test_2d: Any
    x_train_3d: Any
    x_test_3d: Any
    y_train: Any
    y_test: Any