import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([1, 32, 96, 96, 96], dtype=torch.half, device='cuda', requires_grad=True)
net = torch.nn.Conv3d(32, 33, kernel_size=[1, 1, 1], padding=[0, 0, 0], stride=[1, 1, 1], dilation=[1, 1, 1], groups=1)
net = net.cuda().half()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
