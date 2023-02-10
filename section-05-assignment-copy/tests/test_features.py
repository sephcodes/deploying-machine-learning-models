from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    sample_input_data = sample_input_data[0]

    # Given
    transformer = ExtractLetterTransformer(variables=config.model_config.cabin)
    assert sample_input_data["cabin"].iat[6] == "E12"

    subject = transformer.fit_transform(sample_input_data)

    assert subject["cabin"].iat[6] == "E"
