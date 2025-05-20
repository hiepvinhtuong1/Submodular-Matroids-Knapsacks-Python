import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
print(f"Device name: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")