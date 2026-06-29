"""Fixed-shape observation encoding and neural network components."""
from .encoding import EncoderConfig, encode_observation, encode_observation_numpy
__all__ = ["EncoderConfig", "encode_observation", "encode_observation_numpy"]
