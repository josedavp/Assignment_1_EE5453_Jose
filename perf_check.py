import time, math, torch, numpy as np

torch.set_num_threads(torch.get_num_threads())  # use all CPU threads
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('high')

def tflops_gemm(a, b, t_ms):
    # GEMM does 2*m*n*k FLOPs
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    flops = 2 * m * n * k
    return flops / (t_ms * 1e-3) / 1e12

def bench_cuda_matmul(dtype=torch.float32, size=8192, warmup=3, iters=10):
    device = torch.device('cuda', 0)
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    # warmup
    for _ in range(warmup):
        (a @ b).sum().item()
    torch.cuda.synchronize()
    # timed
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    ms = np.median(times)
    return ms, tflops_gemm(a, b, ms)

def bench_cpu_matmul(dtype=torch.float32, size=8192, warmup=1, iters=3):
    device = torch.device('cpu')
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(warmup):
        (a @ b).sum().item()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    ms = np.median(times)
    return ms, tflops_gemm(a, b, ms)

def gbps(bytes_moved, ms):
    return bytes_moved / (ms * 1e-3) / 1e9

def bench_gpu_bandwidth(nbytes=int(2e9), iters=5):
    # device-to-device
    x = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    times=[]
    for _ in range(iters):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        y.copy_(x)  # D2D
        torch.cuda.synchronize()
        t1=time.perf_counter()
        times.append((t1-t0)*1e3)
    d2d_ms = np.median(times)
    d2d_gbps = gbps(nbytes, d2d_ms)

    # host↔device (pageable host)
    h = torch.empty(nbytes, dtype=torch.uint8, device='cpu')
    # H2D
    times=[]
    for _ in range(iters):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        x.copy_(h, non_blocking=False)
        torch.cuda.synchronize()
        t1=time.perf_counter()
        times.append((t1-t0)*1e3)
    h2d_ms = np.median(times)
    h2d_gbps = gbps(nbytes, h2d_ms)
    # D2H
    times=[]
    for _ in range(iters):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        h.copy_(x, non_blocking=False)
        torch.cuda.synchronize()
        t1=time.perf_counter()
        times.append((t1-t0)*1e3)
    d2h_ms = np.median(times)
    d2h_gbps = gbps(nbytes, d2h_ms)

    return {'D2D_GBps': d2d_gbps, 'H2D_GBps': h2d_gbps, 'D2H_GBps': d2h_gbps}

def bench_cpu_bandwidth(nbytes=int(2e9), iters=5):
    # approximate RAM bandwidth via large memcpy using numpy
    a = np.empty(nbytes, dtype=np.uint8)
    b = np.empty_like(a)
    times=[]
    for _ in range(iters):
        t0=time.perf_counter()
        b[:] = a  # memcpy
        t1=time.perf_counter()
        times.append((t1-t0)*1e3)
    ms = np.median(times)
    return gbps(nbytes, ms)

def main():
    if torch.cuda.is_available():
        print('== GPU GEMM ==')
        ms32, t32 = bench_cuda_matmul(torch.float32)
        print(f'FP32: {t32:.2f} TFLOPs (median {ms32:.1f} ms for 8192x8192)')
        # FP16 on CUDA cores (not tensor cores explicitly)
        ms16, t16 = bench_cuda_matmul(torch.float16)
        print(f'FP16: {t16:.2f} TFLOPs (median {ms16:.1f} ms for 8192x8192)')

        print('== GPU Bandwidth ==')
        bw = bench_gpu_bandwidth()
        print(bw)
    else:
        print('No CUDA detected.')

    print('== CPU GEMM ==')
    ms_cpu, t_cpu = bench_cpu_matmul()
    print(f'CPU FP32: {t_cpu:.3f} TFLOPs (median {ms_cpu:.0f} ms for 8192x8192)')

    print('== CPU RAM bandwidth (approx memcpy) ==')
    cpu_bw = bench_cpu_bandwidth()
    print(f'CPU memcpy bandwidth ~ {cpu_bw:.1f} GB/s')

if __name__ == "__main__":
    main()
