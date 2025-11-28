from typing import Optional, Literal, Dict
from pydantic import BaseModel, Field, ConfigDict


class TitanicInput(BaseModel):
    Pclass: int | None = Field(..., example=3)
    Sex: Optional[Literal['male', 'female']] | None = Field(..., example='female')
    Age: float | int | None = Field(..., example=78.5)
    SibSp: int | None = Field(..., example=1)
    Parch: int | None = Field(..., example=0)
    Fare: float | None = Field(..., example=7.9)
    Embarked: Optional[Literal['S', 'C', 'Q']] | None = Field(..., example='C')

    model_config = ConfigDict(extra='ignore')


class TitanicOutput(BaseModel):
    pred: int = Field(...)
    prob: float = Field(...)
    factors: Dict[str, float] | None
