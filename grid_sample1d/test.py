import torch
from grid_sample1d_cuda import forward
import os

# grid_sample1d = forward(padding_mode=True, align_corners=True)
# TODO: 这里有点不知道怎么算。这个应该默认float32就是说8G的tensor应该对应lenth是多少？
# TODO: 下面不再这里调整了，直接在kernel里面调整好了。
var = 128
N = 1
C = 1
L_in = 1024*1024 *256 *var
L_out = 1024 * 1024 * 64
# N*c*in -- N *out
input = torch.ones((1)).cuda()
# input = torch.ones((N, C, L_in)).cuda()
grids = torch.randn((N, L_out)).cuda()

# TODO：看起来还是需要写一个无tensor的接口

# input = torch.randn((16)).cuda()
# grids = torch.randn((1)).cuda()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
    with_stack=True) as prof:
    # # ，i+16，i+24，i+32，i+48，i+48，i+56
    # groups = [dist.new group([i, i+8])for i in range(8)]
    # for i in range(1e):dist.all to all single(torch.view as real(output), torch.view as real(tensor i))
    # output = grid_sample1d(input, grids)
    output = forward(input, grids, True, True) 

print(output)
rank=2
if not os.path.exists("results"):
    os.makedirs("results")
prof.export_chrome_trace("./results/trace_uni_"+str(var)+ ".json")
# prof.export_chrome_trace("./results/trace_base_"+str(var)+ ".json")