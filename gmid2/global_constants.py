import numpy as np


DEBUG__ = True

# path
BENCHMARK_DIR = "/home/junkyul/conda/gmid2/resource/benchmarks"


# numeric constants
EPSILON = 1e-6
ZERO = 1e-300
ALPHA = 1e-4
WEPS = 1e-6     # weight epsilon
WINF = 1e+6     # weight inf
INF = np.inf     # numpy     Inf = inf = infty = Infinity = PINF
NAN = np.NaN     # numpy     nan = NaN = NAN
TOL = 1e-8


# types
TYPE_CHANCE_VAR = 'C'
TYPE_DECISION_VAR = 'D'
TYPE_PROB_FUNC = 'P'
TYPE_UTIL_FUNC = 'U'
TYPE_POLICY_FUNC = 'S'
TYPE_CHANCE_NODE = 'C'
TYPE_DECISION_NODE = 'D'
TYPE_VALUE_NODE = 'U'
TYPE_MESSAGE_NODE = 'M'
TYPE_LIMID_NETWORK = 'LIMID'
TYPE_ID_NETWORK = 'ID'
TYPE_BN_NETWORK = 'BN'
TYPE_MN_NETWORK = 'MN'
TYPE_CHANCE_BLOCK = 'C'
TYPE_DEC_BLOCK = 'D'
TYPE_MSG_FW = "FW"
TYPE_MSG_BW = "BW"

