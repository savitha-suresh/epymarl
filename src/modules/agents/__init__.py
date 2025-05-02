from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .faulty_agent import FaultyAgent
from .transformer_agent import TransformerAgent



REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["rnn_faulty"] = FaultyAgent
REGISTRY["transformer"] = TransformerAgent

