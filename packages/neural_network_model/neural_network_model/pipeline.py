from sklearn.pipeline import Pipeline

from neural_network_model.config import config
from neural_network_model.processing import preprocessors as pp
from neural_network_model import model


# can't include processor to modify target (only training set), bc sklearn limitation
# will use that preprocessor outside of pipeline in train_pipeline
pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.IMAGE_SIZE)),
                ('cnn_model', model.cnn_clf)])
