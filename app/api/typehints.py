from typing import TypedDict


class TrainInfo(TypedDict):
    accuracy: float


class PredictData(TypedDict):
    id: int
    text: str


class Prediction(TypedDict):
    id: int
    label: int
    confidence: int


class TrainData(TypedDict):
    id: int
    text: str
    label: str
