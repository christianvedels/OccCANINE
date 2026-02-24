"""
Performance Benchmark for OccCANINE
Created: 2026-01-26

Purpose: Benchmark inference speed (ms/record) and memory usage (RAM) on test observations.
"""

from histocc import OccCANINE
import pandas as pd
import glob
import time
import psutil
import os
import platform
import torch


def get_system_info():
    """
    Get detailed system information including CPU, GPU, and OS details.
    
    Returns:
        dict: Dictionary containing system information
    """
    sys_info = {}
    
    # Operating System
    sys_info['os'] = platform.system()
    sys_info['os_version'] = platform.version()
    sys_info['os_release'] = platform.release()
    
    # CPU Information
    sys_info['cpu'] = platform.processor()
    sys_info['cpu_count_physical'] = psutil.cpu_count(logical=False)
    sys_info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    sys_info['cpu_freq_current'] = psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
    sys_info['cpu_freq_max'] = psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A'
    
    # Memory Information
    mem = psutil.virtual_memory()
    sys_info['total_ram_gb'] = mem.total / (1024 ** 3)
    sys_info['available_ram_gb'] = mem.available / (1024 ** 3)
    
    # GPU Information
    if torch.cuda.is_available():
        sys_info['gpu_available'] = True
        sys_info['gpu_count'] = torch.cuda.device_count()
        sys_info['gpu_name'] = torch.cuda.get_device_name(0)
        sys_info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        sys_info['cuda_version'] = torch.version.cuda
    else:
        sys_info['gpu_available'] = False
        sys_info['gpu_count'] = 0
        sys_info['gpu_name'] = 'N/A'
        sys_info['gpu_memory_total_gb'] = 'N/A'
        sys_info['cuda_version'] = 'N/A'
    
    # PyTorch Information
    sys_info['pytorch_version'] = torch.__version__
    
    # Python Information
    sys_info['python_version'] = platform.python_version()
    
    return sys_info


def print_system_info(sys_info):
    """
    Print system information in a formatted way.
    
    Args:
        sys_info (dict): Dictionary containing system information
    """
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"\nOperating System:")
    print(f"  OS: {sys_info['os']} {sys_info['os_release']}")
    print(f"  Version: {sys_info['os_version']}")
    
    print(f"\nCPU:")
    print(f"  Processor: {sys_info['cpu']}")
    print(f"  Physical cores: {sys_info['cpu_count_physical']}")
    print(f"  Logical cores: {sys_info['cpu_count_logical']}")
    print(f"  Current frequency: {sys_info['cpu_freq_current']} MHz")
    print(f"  Max frequency: {sys_info['cpu_freq_max']} MHz")
    
    print(f"\nMemory:")
    print(f"  Total RAM: {sys_info['total_ram_gb']:.2f} GB")
    print(f"  Available RAM: {sys_info['available_ram_gb']:.2f} GB")
    
    print(f"\nGPU:")
    if sys_info['gpu_available']:
        print(f"  GPU Available: Yes")
        print(f"  GPU Count: {sys_info['gpu_count']}")
        print(f"  GPU Name: {sys_info['gpu_name']}")
        print(f"  GPU Memory: {sys_info['gpu_memory_total_gb']:.2f} GB")
        print(f"  CUDA Version: {sys_info['cuda_version']}")
    else:
        print(f"  GPU Available: No")
    
    print(f"\nSoftware:")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  PyTorch: {sys_info['pytorch_version']}")
    
    print("="*60 + "\n")


def performance_benchmark(n_obs=10000, data_path="Data/Test_data/*.csv", behavior="good"):
    """
    Benchmark inference speed and memory usage on n_obs test observations.
    
    Args:
        n_obs (int): Number of observations to test (default: 10000)
        data_path (str): Path to test data files
        behavior (str): Prediction behavior - "good" (greedy, slower but better quality) or 
                       "fast" (flat, faster but lower quality). Default: "good"
    
    Returns:
        dict: Performance metrics including ms/record and RAM usage
    """
    # Get and print system information
    sys_info = get_system_info()
    print_system_info(sys_info)
    
    print(f"\n{'='*60}")
    print(f"Performance Benchmark: Testing on {n_obs} observations")
    print(f"Behavior: {behavior}")
    print(f"{'='*60}")
    
    # Load test data
    csv_files = glob.glob(data_path)
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df = df.sample(n=n_obs, random_state=20) if n_obs < df.shape[0] else df
    df = df.reset_index(drop=True)
    
    print(f"Loaded {df.shape[0]} observations")
    
    # Initialize model
    print("Loading model...")
    mod = OccCANINE()
    
    # Get process for memory tracking
    process = psutil.Process(os.getpid())
    
    # Memory before inference
    mem_before = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    
    # Warm-up run (not counted)
    print("Warm-up run...")
    _ = mod.predict(
        df["occ1"].head(100).tolist(),
        df["lang"].head(100).tolist(),
        behavior=behavior,
        deduplicate=False,
        threshold=0.26
    )
    
    # Actual benchmark
    print(f"Running benchmark on {n_obs} observations...")
    start_time = time.time()
    
    predictions = mod.predict(
        df["occ1"].tolist(),
        df["lang"].tolist(),
        behavior=behavior,
        deduplicate=False,
        threshold=0.26
    )
    
    end_time = time.time()
    
    # Memory after inference
    mem_after = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    
    # Calculate metrics
    total_time_sec = end_time - start_time
    total_time_ms = total_time_sec * 1000
    ms_per_record = total_time_ms / n_obs
    records_per_sec = n_obs / total_time_sec
    mem_used = mem_after - mem_before
    
    # Print results to console
    print(f"\n{'='*60}")
    print("Performance Results:")
    print(f"{'='*60}")
    print(f"Total observations: {n_obs:,}")
    print(f"Total time: {total_time_sec:.2f} seconds ({total_time_ms:.2f} ms)")
    print(f"Time per record: {ms_per_record:.2f} ms/record")
    print(f"Throughput: {records_per_sec:.2f} records/second")
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"{'='*60}\n")
    
    # Prepare results dictionary
    results = {
        "n_observations": n_obs,
        "behavior": behavior,
        "total_time_sec": total_time_sec,
        "total_time_ms": total_time_ms,
        "ms_per_record": ms_per_record,
        "records_per_sec": records_per_sec,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_used_mb": mem_used
    }
    
    # Save results to text file
    output_file = f"Project_dissemination/Paper_replication_package/Data/performance_benchmark_results_{behavior}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OccCANINE Performance Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Observations: {n_obs:,}\n")
        f.write(f"Behavior: {behavior}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("System Information:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Operating System: {sys_info['os']} {sys_info['os_release']}\n")
        f.write(f"OS Version: {sys_info['os_version']}\n")
        f.write(f"CPU: {sys_info['cpu']}\n")
        f.write(f"CPU Cores (Physical): {sys_info['cpu_count_physical']}\n")
        f.write(f"CPU Cores (Logical): {sys_info['cpu_count_logical']}\n")
        f.write(f"CPU Frequency: {sys_info['cpu_freq_current']} MHz (Max: {sys_info['cpu_freq_max']} MHz)\n")
        f.write(f"Total RAM: {sys_info['total_ram_gb']:.2f} GB\n")
        if sys_info['gpu_available']:
            f.write(f"GPU: {sys_info['gpu_name']}\n")
            f.write(f"GPU Memory: {sys_info['gpu_memory_total_gb']:.2f} GB\n")
            f.write(f"CUDA Version: {sys_info['cuda_version']}\n")
        else:
            f.write(f"GPU: Not Available\n")
        f.write(f"Python Version: {sys_info['python_version']}\n")
        f.write(f"PyTorch Version: {sys_info['pytorch_version']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("System Information:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Operating System: {sys_info['os']} {sys_info['os_release']}\n")
        f.write(f"OS Version: {sys_info['os_version']}\n")
        f.write(f"CPU: {sys_info['cpu']}\n")
        f.write(f"CPU Cores (Physical): {sys_info['cpu_count_physical']}\n")
        f.write(f"CPU Cores (Logical): {sys_info['cpu_count_logical']}\n")
        f.write(f"CPU Frequency: {sys_info['cpu_freq_current']} MHz (Max: {sys_info['cpu_freq_max']} MHz)\n")
        f.write(f"Total RAM: {sys_info['total_ram_gb']:.2f} GB\n")
        if sys_info['gpu_available']:
            f.write(f"GPU: {sys_info['gpu_name']}\n")
            f.write(f"GPU Memory: {sys_info['gpu_memory_total_gb']:.2f} GB\n")
            f.write(f"CUDA Version: {sys_info['cuda_version']}\n")
        else:
            f.write(f"GPU: Not Available\n")
        f.write(f"Python Version: {sys_info['python_version']}\n")
        f.write(f"PyTorch Version: {sys_info['pytorch_version']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("Inference Speed:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total time: {total_time_sec:.2f} seconds\n")
        f.write(f"  Total time: {total_time_ms:.2f} milliseconds\n")
        f.write(f"  Time per record: {ms_per_record:.2f} ms/record\n")
        f.write(f"  Throughput: {records_per_sec:.2f} records/second\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("Memory Usage:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Memory before inference: {mem_before:.2f} MB\n")
        f.write(f"  Memory after inference: {mem_after:.2f} MB\n")
        f.write(f"  Memory used: {mem_used:.2f} MB\n\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"Results saved to {output_file}")
    
    # Also save as CSV for further analysis
    csv_file = f"Project_dissemination/Paper_replication_package/Data/performance_benchmark_results_{behavior}.csv"
    df_results = pd.DataFrame([results])
    df_results.to_csv(csv_file, index=False)
    print(f"CSV results saved to {csv_file}")
    
    return results


def main(n_obs=10000, data_path="Data/Test_data/*.csv", behavior="good"):
    """
    Main function to run the performance benchmark.
    
    Args:
        n_obs (int): Number of observations to test (default: 10000)
        data_path (str): Path to test data files
        behavior (str): Prediction behavior - "good" or "fast" (default: "good")
    """
    results = performance_benchmark(n_obs=n_obs, data_path=data_path, behavior=behavior)
    return results


if __name__ == "__main__":
    # Test both behaviors
    print("\n" + "#" * 60)
    print("# Testing FAST behavior")
    print("#" * 60)
    main(n_obs=10000, data_path="Data/Test_data/*.csv", behavior="fast")
    
    print("\n" + "#" * 60)
    print("# Testing GOOD behavior")
    print("#" * 60)
    main(n_obs=10000, data_path="Data/Test_data/*.csv", behavior="good")
