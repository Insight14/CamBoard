"""CamBoard model architectures registry."""

from camboard.models.mlp import LandmarkMLP, build_fingerspelling_mlp, build_gesture_mlp
from camboard.models.gru_seq import GRUSequenceClassifier
from camboard.models.cnn1d import CNN1DClassifier
from camboard.models.bigru import BiGRUClassifier
from camboard.models.temporal_transformer import SignVocabTransformer
from camboard.models.math_model import FullMathModel, MathDecoder, MathEncoder

__all__ = [
    "LandmarkMLP",
    "build_gesture_mlp",
    "build_fingerspelling_mlp",
    "GRUSequenceClassifier",
    "CNN1DClassifier",
    "BiGRUClassifier",
    "SignVocabTransformer",
    "FullMathModel",
    "MathEncoder",
    "MathDecoder",
]

