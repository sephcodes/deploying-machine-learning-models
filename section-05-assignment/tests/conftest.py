from typing import Tuple

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset


@pytest.fixture()
def sample_input_data() -> Tuple[pd.DataFrame, pd.Series]:
    # tbh not sure why using load_raw_dataset instead of load_dataset
    # seeing as the cabin column gets fixed to grab first cabin if multiple
    # and we're using this sample input_data to test in test_features
    data = load_raw_dataset(file_name=config.app_config.raw_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    return X_test, y_test
