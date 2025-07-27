import torch
from typing import Optional, Tuple


def get_gpu_info() -> Optional[str]:
    """Gets GPU device name if available.

    Returns:
        str | None: Name of the GPU device if available, None if no GPU is found.
    """
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_properties(0).name
    return gpu_name


def get_gpu_memory_stats() -> Tuple[float, float]:
    """Get GPU memory statistics.
    
    Returns:
        Tuple of (current_memory_gb, max_memory_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
        
    gpu_stats = torch.cuda.get_device_properties(0)
    current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    return current_memory, max_memory


def print_gpu_stats() -> None:
    """Print current GPU statistics."""
    gpu_name = get_gpu_info()
    current_memory, max_memory = get_gpu_memory_stats()
    
    if gpu_name:
        print(f"GPU = {gpu_name}. Max memory = {max_memory} GB.")
        print(f"{current_memory} GB of memory reserved.")
    else:
        print("No GPU available.")


def ensure_gpu_available() -> str:
    """Ensure GPU is available and return GPU name.
    
    Returns:
        str: GPU name
        
    Raises:
        ValueError: If no GPU is available
    """
    gpu_name = get_gpu_info()
    if not gpu_name:
        raise ValueError("No Nvidia GPU found.")
    return gpu_name