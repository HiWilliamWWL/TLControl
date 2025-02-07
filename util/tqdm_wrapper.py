
import os
from tqdm import tqdm as original_tqdm

def tqdm(*args, **kwargs):
    # 检查是否在Slurm环境中
    is_slurm = "SLURM_JOB_ID" in os.environ
    kwargs['disable'] = kwargs.get('disable', is_slurm)
    return original_tqdm(*args, **kwargs)

