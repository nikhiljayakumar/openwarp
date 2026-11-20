from cli.advisor.rules.cpu_offload import CpuOffloadRule
from cli.advisor.rules.weight_precision import WeightPrecisionRule

DEFAULT_RULES = (CpuOffloadRule(), WeightPrecisionRule())
