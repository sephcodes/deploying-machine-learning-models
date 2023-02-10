from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import transform_dataframe


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    validated_data = transform_dataframe(dataframe=input_data)
    relevant_data = validated_data[config.model_config.features].copy()

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=relevant_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return relevant_data, errors


class TitanicInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[Union[str, int]]
    body: Optional[int]
    homes_dest: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicInputSchema]
