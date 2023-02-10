import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset

#this fixture gets called by the other test_ methods
#tox e- test_package runs the tests in /tests folder
@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)
