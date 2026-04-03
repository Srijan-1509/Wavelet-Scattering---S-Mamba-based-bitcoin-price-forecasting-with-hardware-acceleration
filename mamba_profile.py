import torch
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from mamba_predictor import WaveletMambaWrapper, CombinedLoss

print("[*] Starting fast check")
device = torch.device('cuda')
print("[*] Building Mamba...")
builder = WaveletMambaWrapper(window_size=64, n_features=9)
builder.build_model()
model = builder.model
model.train()
print("[*] Model built")

X = torch.randn(128, 64, 9)
y_p = torch.randn(128, 1)
y_d = torch.randint(0, 2, (128, 1)).float()

dataset = TensorDataset(X, y_p, y_d)
loader = DataLoader(dataset, batch_size=128)

criterion = CombinedLoss()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler('cuda')

print("[*] Starting loop")
for i, (xb, yp, yd) in enumerate(loader):
    print(f"Batch {i}: loading to device")
    t0 = time.time()
    xb = xb.to(device)
    yp = yp.to(device)
    yd = yd.to(device)
    torch.cuda.synchronize()
    print(f" -> {time.time()-t0:.4f}s")
    
    print(f"Batch {i}: forward pass")
    t0 = time.time()
    with autocast(device_type='cuda', dtype=torch.float16):
        out_p, out_d = model(xb)
        loss, _, _ = criterion(out_p, out_d, yp, yd)
    torch.cuda.synchronize()
    print(f" -> {time.time()-t0:.4f}s")
    
    print(f"Batch {i}: backward pass")
    t0 = time.time()
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    print(f" -> {time.time()-t0:.4f}s")
    
    print(f"Batch {i}: optimizer step")
    t0 = time.time()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f" -> {time.time()-t0:.4f}s")
    
    print("[OK] Done!")
    break
