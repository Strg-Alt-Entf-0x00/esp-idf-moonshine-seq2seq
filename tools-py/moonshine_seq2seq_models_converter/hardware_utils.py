#!/usr/bin/env python3
"""
Hardware Detection and Optimization Utilities

Automatically detects CPU vendor, GPU availability, and applies
optimal TensorFlow/PyTorch configurations for model conversion.

Supports:
- Intel CPU (oneDNN enabled)
- AMD CPU (oneDNN disabled, AMD optimizations)
- ARM CPU (Apple Silicon, Linux ARM)
- NVIDIA GPU (CUDA/cuDNN)
- AMD GPU (ROCm, experimental)
- Apple GPU (Metal)

Usage:
    from hardware_utils import setup_hardware, get_hardware_info
    
    # Automatic detection
    device = setup_hardware()
    
    # Manual override
    device = setup_hardware(device='gpu', gpu_id=0)
    
    # Just detect, don't configure
    info = get_hardware_info()
"""

import os
import sys
import platform
import logging

logger = logging.getLogger(__name__)


def get_cpu_vendor():
    """
    Detect CPU vendor (Intel, AMD, ARM, Apple, Unknown)
    
    Returns:
        str: 'intel', 'amd', 'arm', 'apple', or 'unknown'
    """
    system = platform.system()
    
    # Apple Silicon
    if system == 'Darwin' and platform.machine() == 'arm64':
        return 'apple'
    
    # Try cpuinfo library first (most reliable)
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        vendor = info.get('vendor_id_raw', '').lower()
        brand = info.get('brand_raw', '').lower()
        
        if 'intel' in vendor or 'intel' in brand:
            return 'intel'
        elif 'amd' in vendor or 'advanced micro devices' in vendor or 'amd' in brand:
            return 'amd'
        elif 'arm' in vendor or 'arm' in brand:
            return 'arm'
        elif 'apple' in vendor or 'apple' in brand:
            return 'apple'
    except ImportError:
        pass
    
    # Fallback: platform module
    processor = platform.processor().lower()
    machine = platform.machine().lower()
    
    if 'intel' in processor or 'i3' in processor or 'i5' in processor or 'i7' in processor or 'i9' in processor:
        return 'intel'
    elif 'amd' in processor or 'ryzen' in processor or 'epyc' in processor:
        return 'amd'
    elif 'arm' in processor or 'arm' in machine or 'aarch' in machine:
        return 'arm'
    
    # Last resort: check architecture
    if machine in ['amd64', 'x86_64']:
        # Can't distinguish Intel vs AMD without additional info
        return 'unknown'
    elif 'arm' in machine or 'aarch' in machine:
        return 'arm'
    
    return 'unknown'


def get_cpu_count():
    """
    Get number of CPU cores (physical + logical)
    
    Returns:
        dict: {'physical': int, 'logical': int}
    """
    try:
        import psutil
        return {
            'physical': psutil.cpu_count(logical=False),
            'logical': psutil.cpu_count(logical=True)
        }
    except ImportError:
        logical = os.cpu_count() or 1
        return {
            'physical': logical // 2,  # Rough estimate
            'logical': logical
        }


def detect_nvidia_gpu():
    """
    Detect NVIDIA GPU using multiple methods
    
    Returns:
        list: List of GPU info dicts or empty list if none found
    """
    gpus = []
    
    # Method 1: TensorFlow
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            for i, gpu in enumerate(tf_gpus):
                gpus.append({
                    'id': i,
                    'name': gpu.name,
                    'type': 'nvidia',
                    'method': 'tensorflow'
                })
            return gpus
    except Exception:
        pass
    
    # Method 2: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpus.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2]),
                            'type': 'nvidia',
                            'method': 'nvidia-smi'
                        })
            return gpus
    except Exception:
        pass
    
    # Method 3: pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                'id': i,
                'name': name.decode('utf-8') if isinstance(name, bytes) else name,
                'memory_mb': memory.total // (1024 * 1024),
                'type': 'nvidia',
                'method': 'pynvml'
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass
    
    return gpus


def get_hardware_info():
    """
    Comprehensive hardware detection
    
    Returns:
        dict: Complete hardware information
    """
    cpu_vendor = get_cpu_vendor()
    cpu_count = get_cpu_count()
    gpus = detect_nvidia_gpu()
    
    info = {
        'cpu': {
            'vendor': cpu_vendor,
            'cores_physical': cpu_count['physical'],
            'cores_logical': cpu_count['logical'],
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'gpu': {
            'available': len(gpus) > 0,
            'count': len(gpus),
            'devices': gpus
        },
        'system': {
            'os': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        }
    }
    
    # Determine recommended device
    if info['gpu']['available']:
        info['recommended_device'] = 'gpu'
        info['recommended_reason'] = f"GPU acceleration available ({info['gpu']['count']} device(s))"
    else:
        info['recommended_device'] = 'cpu'
        info['recommended_reason'] = "No GPU detected"
    
    # Determine optimizations
    optimizations = []
    if cpu_vendor == 'intel':
        optimizations.append('oneDNN')
    if info['gpu']['available']:
        optimizations.append('CUDA/cuDNN')
    
    info['optimizations'] = optimizations
    
    return info


def setup_tensorflow_gpu(gpu_id=None, memory_growth=True):
    """
    Configure TensorFlow for GPU usage
    
    Args:
        gpu_id: Specific GPU to use (None = all GPUs)
        memory_growth: Enable memory growth to avoid OOM
    
    Returns:
        bool: True if GPU setup successful
    """
    try:
        import tensorflow as tf
        
        # Check for Windows GPU limitation
        if platform.system() == 'Windows':
            import tensorflow as tf
            tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
            if tf_version >= (2, 11):
                logger.warning(
                    "TensorFlow 2.11+ does not support native GPU on Windows.\n"
                    "  Options:\n"
                    "    1. Use WSL2 (recommended): https://docs.microsoft.com/en-us/windows/wsl/install\n"
                    "    2. Install tensorflow-directml: pip install tensorflow-directml\n"
                    "    3. Downgrade TensorFlow: pip install tensorflow==2.10\n"
                    "  Falling back to CPU for now."
                )
                return False
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs found by TensorFlow")
            return False
        
        if gpu_id is not None:
            # Use specific GPU
            if gpu_id >= len(gpus):
                logger.error(f"GPU {gpu_id} not found (only {len(gpus)} available)")
                return False
            gpus = [gpus[gpu_id]]
            tf.config.set_visible_devices(gpus, 'GPU')
        
        # Enable memory growth
        if memory_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth: {e}")
        
        logger.info(f"✓ TensorFlow GPU setup successful ({len(gpus)} device(s))")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup TensorFlow GPU: {e}")
        return False


def setup_cpu_optimizations(cpu_vendor):
    """
    Configure CPU-specific optimizations
    
    Args:
        cpu_vendor: CPU vendor ('intel', 'amd', 'arm', etc.)
    """
    if cpu_vendor == 'intel':
        # Enable oneDNN for Intel CPUs (significant speedup)
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        logger.info("✓ Intel CPU detected → oneDNN enabled")
    else:
        # Disable oneDNN for non-Intel CPUs (can be slower)
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        if cpu_vendor == 'amd':
            logger.info("✓ AMD CPU detected → oneDNN disabled")
        elif cpu_vendor == 'arm':
            logger.info("✓ ARM CPU detected → oneDNN disabled")
        elif cpu_vendor == 'apple':
            logger.info("✓ Apple Silicon detected → oneDNN disabled")
        else:
            logger.info("✓ Unknown CPU → oneDNN disabled (safe default)")
    
    # Set TensorFlow logging level
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_hardware(device='auto', gpu_id=None, num_threads=None, verbose=True):
    """
    Automatically detect and configure optimal hardware settings using PyTorch
    
    Args:
        device: 'auto', 'cpu', or 'gpu'
        gpu_id: Specific GPU to use (only for device='gpu')
        num_threads: Number of CPU threads (None = auto)
        verbose: Print hardware information
    
    Returns:
        str: PyTorch device string ('cuda', 'cuda:0', or 'cpu')
    """
    # Get hardware info
    hw_info = get_hardware_info()
    
    # Check PyTorch CUDA availability
    try:
        import torch
        pytorch_cuda = torch.cuda.is_available()
        pytorch_gpu_count = torch.cuda.device_count() if pytorch_cuda else 0
    except:
        pytorch_cuda = False
        pytorch_gpu_count = 0
    
    # Resolve device
    if device == 'auto':
        device = 'gpu' if pytorch_cuda else 'cpu'
    
    # Validate device choice
    if device == 'gpu' and not pytorch_cuda:
        logger.error("GPU requested but PyTorch CUDA not available! Falling back to CPU")
        device = 'cpu'
    
    # Print hardware summary
    if verbose:
        print("=" * 80)
        print("HARDWARE CONFIGURATION")
        print("=" * 80)
        print(f"CPU:     {hw_info['cpu']['vendor'].upper()} "
              f"({hw_info['cpu']['cores_physical']} cores / "
              f"{hw_info['cpu']['cores_logical']} threads)")
        
        if pytorch_cuda:
            for i in range(pytorch_gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory // (1024**2)
                print(f"GPU {i}:   {gpu_name} ({gpu_mem} MB)")
        else:
            print("GPU:     None detected (PyTorch CUDA not available)")
        
        print(f"\nUsing:   {device.upper()}")
        
        if device == 'gpu':
            print(f"Framework: PyTorch CUDA (native Windows support)")
        
        print("=" * 80)
        print()
    
    # Configure based on device
    if device == 'gpu':
        import torch
        if gpu_id is not None:
            torch_device = f'cuda:{gpu_id}'
        else:
            torch_device = 'cuda'
        
        # Test GPU
        try:
            test_tensor = torch.zeros(1).to(torch_device)
            logger.info(f"✓ PyTorch GPU ready: {torch_device}")
            return torch_device
        except Exception as e:
            logger.error(f"GPU setup failed: {e}, falling back to CPU")
            device = 'cpu'
    
    if device == 'cpu':
        setup_cpu_optimizations(hw_info['cpu']['vendor'])
        
        # Set CPU threads for PyTorch
        if num_threads is not None and num_threads > 0:
            try:
                import torch
                torch.set_num_threads(num_threads)
                logger.info(f"✓ PyTorch using {num_threads} CPU threads")
            except Exception as e:
                logger.warning(f"Could not set thread count: {e}")
    
    return 'cpu'


if __name__ == "__main__":
    # Test/demo mode
    print("Hardware Detection Demo")
    print("=" * 80)
    
    info = get_hardware_info()
    
    print("\nCPU Information:")
    print(f"  Vendor:   {info['cpu']['vendor']}")
    print(f"  Cores:    {info['cpu']['cores_physical']} physical / {info['cpu']['cores_logical']} logical")
    print(f"  Platform: {info['cpu']['platform']}")
    print(f"  Machine:  {info['cpu']['machine']}")
    
    print("\nGPU Information:")
    print(f"  Available: {info['gpu']['available']}")
    print(f"  Count:     {info['gpu']['count']}")
    for gpu in info['gpu']['devices']:
        memory = f" ({gpu['memory_mb']} MB)" if 'memory_mb' in gpu else ""
        print(f"  Device {gpu['id']}: {gpu['name']}{memory}")
    
    print("\nRecommendation:")
    print(f"  Device: {info['recommended_device']}")
    print(f"  Reason: {info['recommended_reason']}")
    print(f"  Optimizations: {', '.join(info['optimizations']) if info['optimizations'] else 'None'}")
    
    print("\n" + "=" * 80)
    print("Testing setup_hardware()...")
    print("=" * 80)
    device = setup_hardware(device='auto', verbose=True)
    print(f"\nFinal device: {device}")
