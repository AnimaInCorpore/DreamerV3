import json
import platform
import sys
import sysconfig
import torch

result = dict(python=sys.version, executable=sys.executable, platform=sys.platform,
              architecture=sysconfig.get_platform(), system=platform.platform(),
              torch=torch.__version__, cuda_runtime=torch.version.cuda,
              cuda_available=torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    x = torch.randn(128,128,device='cuda')
    loss = (x @ x.T).square().mean()
    torch.cuda.synchronize()
    result.update(gpu=props.name, compute_capability=[props.major,props.minor],
                  vram_bytes=props.total_memory, free_vram_bytes=torch.cuda.mem_get_info()[0],
                  cuda_matrix_test=float(loss), pytorch_architectures=torch.cuda.get_arch_list())
print(json.dumps(result,indent=2))
