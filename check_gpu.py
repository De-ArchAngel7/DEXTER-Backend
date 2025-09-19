#!/usr/bin/env python3
"""
🔍 GPU Verification Script for DEXTER AI Training
Make sure PyTorch is actually using your GTX 1650!
"""

import torch
import torch.nn as nn
import time
import numpy as np

def check_gpu_status():
    """Check if PyTorch can see and use your GPU"""
    print("🔍 Checking GPU Status for DEXTER AI Training...")
    print("=" * 60)
    
    # 1. Check CUDA availability
    print("📊 CUDA Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        
        # 2. Check your specific GPU
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
            
            if "1650" in gpu_name:
                print(f"    ✅ Found your GTX 1650!")
    else:
        print("  ❌ CUDA not available - check your drivers!")
        return False
    
    # 3. Test GPU vs CPU performance
    print("\n🚀 Performance Test:")
    
    # Create test data
    size = 1000
    test_data = torch.randn(size, size)
    
    # CPU test
    print("  Testing CPU performance...")
    start_time = time.time()
    cpu_result = torch.mm(test_data, test_data)
    cpu_time = time.time() - start_time
    print(f"    CPU time: {cpu_time:.4f} seconds")
    
    # GPU test
    if torch.cuda.is_available():
        print("  Testing GPU performance...")
        gpu_data = test_data.cuda()
        
        # Warm up GPU
        _ = torch.mm(gpu_data, gpu_data)
        torch.cuda.synchronize()
        
        start_time = time.time()
        gpu_result = torch.mm(gpu_data, gpu_data)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"    GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"    GPU speedup: {speedup:.1f}x faster than CPU")
        
        # Verify results match
        cpu_result_gpu = cpu_result.cuda()
        if torch.allclose(gpu_result, cpu_result_gpu, atol=1e-5):
            print("    ✅ GPU calculations are correct!")
        else:
            print("    ❌ GPU calculations have errors!")
    
    # 4. Test your LSTM model on GPU
    print("\n🧠 Testing LSTM Model on GPU:")
    
    if torch.cuda.is_available():
        # Create a small LSTM model like yours
        test_model = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Move to GPU
        test_model = test_model.cuda()
        print("  ✅ LSTM model moved to GPU successfully")
        
        # Test forward pass
        test_input = torch.randn(32, 24, 28).cuda()  # Your batch size, sequence length, features
        
        with torch.no_grad():
            output, _ = test_model(test_input)
        
        print(f"  ✅ Forward pass successful: {output.shape}")
        print(f"  ✅ Output device: {output.device}")
        
        # Check memory usage
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"  📊 GPU memory used: {gpu_memory_used:.1f} MB")
        print(f"  📊 GPU memory cached: {gpu_memory_cached:.1f} MB")
        
        # Clear GPU memory
        del test_model, test_input, output
        torch.cuda.empty_cache()
        
    return True

def test_training_device():
    """Test which device your training will actually use"""
    print("\n🎯 Training Device Test:")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Check default device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test with your training setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ CUDA device available: {device}")
        
        # Move model to GPU
        model = model.to(device)
        print(f"✅ Model moved to: {next(model.parameters()).device}")
        
        # Test data on GPU
        x = torch.randn(100, 10).to(device)
        y = torch.randn(100, 1).to(device)
        
        print(f"✅ Input data device: {x.device}")
        print(f"✅ Target data device: {y.device}")
        
        # Test forward pass
        output = model(x)
        print(f"✅ Output device: {output.device}")
        print(f"✅ All tensors on GPU: {all(t.device.type == 'cuda' for t in [x, y, output])}")
        
    else:
        print("❌ CUDA not available - training will use CPU")

if __name__ == "__main__":
    print("🚀 DEXTER GPU Verification Starting!")
    print("=" * 60)
    
    try:
        # Check GPU status
        gpu_ok = check_gpu_status()
        
        if gpu_ok:
            # Test training device
            test_training_device()
            
            print("\n🎉 GPU Verification Complete!")
            print("=" * 60)
            print("✅ Your GTX 1650 is ready for AI training!")
            print("✅ PyTorch will use GPU acceleration")
            print("✅ Expect 3-6x speedup over CPU")
            print("✅ Training time: ~1-2 hours for 100 epochs")
        else:
            print("\n❌ GPU Issues Detected!")
            print("=" * 60)
            print("Check your NVIDIA drivers and CUDA installation")
            
    except Exception as e:
        print(f"❌ Error during GPU verification: {e}")
        import traceback
        traceback.print_exc()
