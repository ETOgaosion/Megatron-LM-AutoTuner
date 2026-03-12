"""
This file shows a minimal example of using NVSHMEM4Py to run a collective operation on CuPy arrays.
This example demonstrates direct GPU-to-GPU communication using NVSHMEM's symmetric memory model,
showing how to perform point-to-point operations between NVLink-Accesible PEs (Processing Elements)
using the nvshmem.core.get_peer_array() function.
"""

import cupy
import nvshmem.core
from cuda.core.experimental import Device, system
from numba import cuda
import mpi4py.MPI as MPI

@cuda.jit
def simple_shift(arr, dst_pe):
    """
    CUDA kernel that performs a simple point-to-point communication.
    Writes the destination PE's ID directly to the array on the target GPU.
    This operation uses NVLink for direct GPU-to-GPU communication if the destination PE is in the same NVLink domain.
    The array passed in should be retrieved using nvshmem.core.get_peer_array() which returns an Array on the symmetric heap that points to another PE's memory.
    """
    # This line is issued as an NVLink Store to the destination PE
    arr[0] = dst_pe

# Initialize NVSHMEM Using an MPI communicator
# Calculate local rank within the node to determine which GPU to use
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()  # Set the current CUDA device
stream = dev.create_stream()  # Create CUDA stream for asynchronous operations

# Initialize NVSHMEM with MPI communicator
nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

# Create a symmetric array that is accessible from all PEs
# This array will be used for point-to-point communication
array = nvshmem.core.array((1,), dtype="int32")

# Get current PE ID and calculate destination PE for the ring communication
my_pe = nvshmem.core.my_pe()
# A unidirectional ring - always get the neighbor to the right
dst_pe = (my_pe + 1) % nvshmem.core.n_pes()

# Get a view of the destination PE's array for direct access
# This enables direct GPU-to-GPU communication over NVLink
# Note: The destination PE must be in the same NVLink domain
# If it's not accessible, this will raise an Exception
dev_dst = nvshmem.core.get_peer_array(b, dst_pe)

# Launch the CUDA kernel to perform the point-to-point communication
block = 1
grid = (size + block - 1) // block
simple_shift[block, grid, 0, 0](array, my_pe)

# Synchronize all PEs in the node to ensure communication is complete
nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream)

# Print the result - should show the value written by the neighboring PE
print(f"From PE {my_pe}, array contains {array}")

# Clean up NVSHMEM resources
nvshmem.core.free_array(arr_src)
nvshmem.core.free_array(arr_dst)
nvshmem.core.finalize()  # Finalize NVSHMEM