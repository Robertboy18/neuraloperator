import socket
from datetime import datetime
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfileTimeline
import torch

categories = {"PARAMETER": 0,
              "OPT": 1,
              "INPUT": 2,
              "TEMP": 3,
              "ACTIVATION": 4,
              "GRADS": 5,
              "AUTOGRAD_DETAIL": 6,
              "None": 7}
GB = 1024**3
device_str = "cuda:0"
def trace_handler(prof):
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"trace_{timestamp}_{host_name}"

    # export raw memory timeline
    mem_tl = MemoryProfileTimeline(prof._memory_profile())
    times, sizes = mem_tl._coalesce_timeline(device_str)
    times = np.array(times)
    sizes = np.array(sizes)

    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / GB
    device = torch.device(device_str)

    msg= f"Memory Timeline for {device_str}\n"
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)
    msg += f"Max CUDA allocated (GB): {max_memory_allocated / GB :.2f}\n"
    msg += f"Max CUDA reserved (GB): {max_memory_reserved / GB :.2f}\n"

    opt_mems = sizes[:,categories["OPT"]+1]
    max_opt_mem = np.max(opt_mems)
    msg += f"Max optimizer state (GB): {max_opt_mem / GB :.2f}\n"
    
    grad_mems = sizes[:,categories["GRADS"]+1]
    max_grad_mem = np.max(grad_mems)
    msg += f"Max gradient memory (GB): {max_grad_mem / GB :.2f}\n"

    param_mems = sizes[:, categories["PARAMETER"]+1]
    max_param_mem = np.max(param_mems)
    msg += f"Max parameter memory (GB): {max_param_mem / GB :.2f}\n"


    with open(f"./mem_stats/{file_prefix}.txt", "w") as f:
        f.write(msg)
    f.close()

    # Construct the memory timeline file.
    fname = f"./snapshots/{file_prefix}.raw.json.gz"
    prof.export_memory_timeline(fname, device=device_str)