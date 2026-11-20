from cli.advisor.rules.cpu_offload import CpuOffloadRule
from cli.advisor.rules.kv_fragmentation import KvFragmentationRule

DEFAULT_RULES = (CpuOffloadRule(), KvFragmentationRule())
