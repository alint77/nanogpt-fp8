"""
Optimized DataLoader for OpenWebText binary files.
Features:
- Efficient memory-mapped file access
- Pre-allocated tensors to avoid repeated allocations
- Proper pinned memory for async GPU transfers
- Support for multi-process data loading
- Compatible with DDP training
"""

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class OpenWebTextDataset(IterableDataset):
    """
    Memory-efficient iterable dataset that streams from binary files.
    Each worker gets a different random seed for true randomness in multi-worker scenarios.
    """
    
    def __init__(self, data_path, block_size, split='train', seed=1337):
        super().__init__()
        self.data_path = data_path
        self.block_size = block_size
        self.split = split
        self.seed = seed
        
        # Load memmap to get length, then close
        data_file = os.path.join(data_path, f'{split}.bin')
        temp_data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.data_len = len(temp_data)
        del temp_data  # Close the memmap
        
        # We'll create a new memmap in __iter__ for each worker
        self.data = None
        
    def __iter__(self):
        # Get worker info for proper seeding in multi-worker setup
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multiple workers: each gets unique seed
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Seed based on base seed, worker id, and process id (for DDP)
            seed = self.seed + worker_id
        else:
            # Single worker
            seed = self.seed
            
        # Set seed for this worker
        rng = np.random.default_rng(seed)
        
        # Create memmap for this worker
        data_file = os.path.join(self.data_path, f'{self.split}.bin')
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        
        # Infinite iterator - yields random samples forever
        while True:
            # Random starting position
            start_idx = rng.integers(0, self.data_len - self.block_size - 1)
            
            # Load sequences efficiently as contiguous numpy arrays
            x = self.data[start_idx:start_idx + self.block_size].astype(np.int64)
            y = self.data[start_idx + 1:start_idx + self.block_size + 1].astype(np.int64)
            
            # Convert to torch tensors
            yield torch.from_numpy(x), torch.from_numpy(y)


def create_dataloader(data_path, block_size, batch_size, split='train', 
                      num_workers=4, seed=1337, pin_memory=True, prefetch_factor=2):
    """
    Create an optimized DataLoader for training.
    
    Args:
        data_path: Path to directory containing train.bin and val.bin
        block_size: Sequence length
        batch_size: Batch size
        split: 'train' or 'val'
        num_workers: Number of background workers for data loading
                     4 is a good default for most systems
        seed: Random seed
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        prefetch_factor: How many batches to prefetch per worker
        
    Returns:
        DataLoader instance
    """
    
    dataset = OpenWebTextDataset(
        data_path=data_path,
        block_size=block_size,
        split=split,
        seed=seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        drop_last=False,  # Not needed for infinite dataset
    )
    
    return loader


class FastDataLoader:
    """
    Ultra-fast single-process dataloader with pre-allocated tensors.
    Best for when you don't need background workers (already fast enough).
    Avoids Python GIL overhead and multiprocessing complexity.
    """
    
    def __init__(self, data_path, block_size, batch_size, split='train', 
                 seed=1337, device='cuda', rank=0):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        self.device = device
        
        # Load memmap
        data_file = os.path.join(data_path, f'{split}.bin')
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.data_len = len(self.data)
        
        # Pre-allocate pinned memory tensors for zero-copy transfers
        self.x_pin = torch.empty((batch_size, block_size), 
                                 dtype=torch.long, pin_memory=True)
        self.y_pin = torch.empty((batch_size, block_size), 
                                 dtype=torch.long, pin_memory=True)
        
        # Pre-allocate device tensors (optional, for even faster transfers)
        if device != 'cpu':
            self.x_device = torch.empty((batch_size, block_size), 
                                       dtype=torch.long, device=device)
            self.y_device = torch.empty((batch_size, block_size), 
                                       dtype=torch.long, device=device)
        else:
            self.x_device = self.x_pin
            self.y_device = self.y_pin
        
        # Random number generator (seeded per rank for DDP)
        self.rng = np.random.default_rng(seed + rank)
        
    def get_batch(self):
        """
        Get a single batch. Optimized for minimal overhead.
        Returns tensors on the target device.
        """
        # Generate random indices
        indices = self.rng.integers(0, self.data_len - self.block_size - 1, 
                                    size=self.batch_size)
        
        # Load data directly into numpy arrays (contiguous memory)
        for i, idx in enumerate(indices):
            # Use numpy's efficient slicing and copy into pre-allocated tensor
            x_np = self.data[idx:idx + self.block_size].astype(np.int64)
            y_np = self.data[idx + 1:idx + self.block_size + 1].astype(np.int64)
            
            # Copy into pinned memory tensors
            self.x_pin[i].copy_(torch.from_numpy(x_np))
            self.y_pin[i].copy_(torch.from_numpy(y_np))
        
        # Async copy to device (non-blocking)
        if self.device != 'cpu':
            self.x_device.copy_(self.x_pin, non_blocking=True)
            self.y_device.copy_(self.y_pin, non_blocking=True)
            return self.x_device, self.y_device
        else:
            return self.x_pin, self.y_pin
    
    def __iter__(self):
        """Make it iterable for compatibility."""
        while True:
            yield self.get_batch()


class VectorizedFastDataLoader:
    """
    Even faster dataloader using vectorized operations.
    Avoids Python loops entirely for maximum performance.
    """
    
    def __init__(self, data_path, block_size, batch_size, split='train', 
                 seed=1337, device='cuda', rank=0):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        self.device = device
        
        # Load memmap
        data_file = os.path.join(data_path, f'{split}.bin')
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.data_len = len(self.data)
        
        # Pre-allocate tensors
        self.x_cpu = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
        self.y_cpu = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
        
        # Random generator
        self.rng = np.random.default_rng(seed + rank)
        
        # Pre-compute offset array for vectorized indexing
        self.offsets = torch.arange(block_size, dtype=torch.long)
        
    def get_batch(self):
        """
        Vectorized batch loading - fastest implementation.
        """
        # Random starting indices for each sequence in batch
        start_indices = self.rng.integers(0, self.data_len - self.block_size - 1, 
                                          size=self.batch_size)
        
        # Vectorized loading using fancy indexing
        # Create index array: (batch_size, block_size)
        idx_x = start_indices[:, None] + self.offsets.numpy()
        idx_y = idx_x + 1
        
        # Fancy indexing to load all data at once
        x_np = self.data[idx_x].astype(np.int64)
        y_np = self.data[idx_y].astype(np.int64)
        
        # Convert to torch (zero-copy with from_numpy, then copy to pinned)
        self.x_cpu.copy_(torch.from_numpy(x_np))
        self.y_cpu.copy_(torch.from_numpy(y_np))
        
        # Move to device
        if self.device != 'cpu':
            return (self.x_cpu.to(self.device, non_blocking=True),
                    self.y_cpu.to(self.device, non_blocking=True))
        else:
            return self.x_cpu, self.y_cpu
    
    def __iter__(self):
        """Make it iterable."""
        while True:
            yield self.get_batch()


# Example usage:
if __name__ == '__main__':
    # Test the dataloaders
    data_path = 'data/openwebtext-1M'
    block_size = 1024
    batch_size = 24
    
    print("Testing VectorizedFastDataLoader...")
    import time
    
    loader = VectorizedFastDataLoader(
        data_path=data_path,
        block_size=block_size,
        batch_size=batch_size,
        split='train',
        device='cuda'
    )
    
    # Warmup
    for _ in range(10):
        x, y = loader.get_batch()
    
    # Benchmark
    num_batches = 100
    start = time.time()
    for _ in range(num_batches):
        x, y = loader.get_batch()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Loaded {num_batches} batches in {elapsed:.3f}s")
    print(f"Average: {elapsed/num_batches*1000:.2f}ms per batch")
    print(f"Throughput: {batch_size * block_size * num_batches / elapsed / 1e6:.2f}M tokens/sec")
    print(f"Batch shape: x={x.shape}, y={y.shape}")
    print(f"Device: {x.device}")
