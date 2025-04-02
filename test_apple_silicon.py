#!/usr/bin/env python3
"""
Test script to check Apple Silicon and MPS (Metal Performance Shaders) support.
This helps verify that PyTorch is properly configured for Apple Silicon.
"""

import sys
import platform
import torch
import numpy as np
from time import time

def check_hardware():
    """Check hardware capabilities"""
    print("==== Hardware Information ====")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    # Check for Apple Silicon
    is_arm = platform.machine() == 'arm64' or 'Apple M' in platform.processor()
    print(f"Is Apple Silicon: {is_arm}")
    
    print("\n==== PyTorch Configuration ====")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS (Metal Performance Shaders) availability
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    if mps_available:
        print(f"MPS device count: {torch.backends.mps.device_count()}")
        print(f"MPS current device: {torch.backends.mps.current_device()}")
    
    return mps_available

def run_performance_test(use_mps=True):
    """Run a simple performance test to compare CPU vs MPS"""
    print("\n==== Performance Test ====")
    
    # Create test tensors
    size = 2000
    print(f"Creating random tensors of size {size}x{size}...")
    
    # CPU tensor
    start = time()
    a_cpu = torch.rand(size, size)
    b_cpu = torch.rand(size, size)
    cpu_create_time = time() - start
    
    # Run CPU matmul
    start = time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_compute_time = time() - start
    
    print(f"CPU tensor creation time: {cpu_create_time:.4f}s")
    print(f"CPU matrix multiplication time: {cpu_compute_time:.4f}s")
    
    # Only run MPS test if available and requested
    if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS tensor
        start = time()
        a_mps = torch.rand(size, size, device="mps")
        b_mps = torch.rand(size, size, device="mps")
        mps_create_time = time() - start
        
        # Run MPS matmul
        start = time()
        c_mps = torch.matmul(a_mps, b_mps)
        # Add sync to ensure timing is fair
        torch.mps.synchronize()
        mps_compute_time = time() - start
        
        print(f"MPS tensor creation time: {mps_create_time:.4f}s")
        print(f"MPS matrix multiplication time: {mps_compute_time:.4f}s")
        
        # Calculate speedup
        if cpu_compute_time > 0:
            speedup = cpu_compute_time / mps_compute_time
            print(f"MPS speedup over CPU: {speedup:.2f}x")
        
        # Verify results match
        c_mps_cpu = c_mps.cpu()
        error = torch.abs(c_cpu - c_mps_cpu).max().item()
        print(f"Maximum numerical error between CPU and MPS: {error:.6e}")
        
        return True
    else:
        print("MPS test skipped - not available or not requested")
        return False

if __name__ == "__main__":
    print("Testing Apple Silicon and MPS support...")
    mps_available = check_hardware()
    
    if mps_available:
        print("\nMPS is available! Running performance test...")
        run_performance_test(use_mps=True)
        print("\n✅ Your system supports Apple Silicon acceleration for PyTorch!")
        print("The Image Style Transfer app should use MPS for faster processing.")
    else:
        print("\n⚠️ MPS is not available on your system.")
        print("The Image Style Transfer app will fall back to CPU processing.")
        print("If you're on Apple Silicon, make sure you have the correct PyTorch version installed:")
        print("pip install torch torchvision")
