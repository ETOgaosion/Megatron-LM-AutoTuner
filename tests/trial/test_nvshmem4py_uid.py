"""
This example shows how to initialize NVSHMEM4Py using an NVSHMEM Unique ID.

In this example, MPI4Py is used to perform a broadcast to share the Unique ID object across all processes.

To initialize NVSHMEM4Py with a custom launcher, you can use a similar approach, replacing the MPI4Py broadcast with data movement handled by your custom launcher.
"""

import mpi4py.MPI as MPI
import nvshmem.core
import numpy as np
from cuda.core.experimental import Device, system

# Use MPI4Py to retrieve the MPI communicator and rank information
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# Find a Device which is unique to each rank
local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
dev = Device(local_rank_per_node)
dev.set_current()

# Create an empty uniqueid for all ranks
uniqueid = nvshmem.core.get_unique_id(empty=True)
if rank == 0:
    # Rank 0 gets a populated uniqueid
    uniqueid = nvshmem.core.get_unique_id()

# Broadcast UID to all ranks
# This is what the custom launcher would need to do if you want to avoid using MPI4Py
comm.Bcast(uniqueid._data.view(np.int8), root=0)

# NVSHMEM Processing Elements (PEs) are bound to a specific device at init time 
nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks,
                  initializer_method="uid")

# Do your NVSHMEM work here

nvshmem.core.finalize()