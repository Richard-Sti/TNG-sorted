# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Script to select particles within a sub-box of a simulation snapshot.
"""
from datetime import datetime
from gc import collect

import MAS_library as MASL
import numpy
from h5py import File
from numba import jit

from tngsorted import get_snapshot_files


@jit(nopython=True, fastmath=True, boundscheck=False)
def pbc_distance(x1, x2, boxsize):
    """Calculate periodic distance between two points."""
    delta = abs(x1 - x2)
    return min(delta, boxsize - delta)


@jit(nopython=True, fastmath=True, boundscheck=False)
def find_next_particle(start_index, end_index, pos, x0, y0, z0,
                       half_width, boxsize):
    """
    Find the next particle in a box of size `half_width` centered on `x0`,
    `y0`, `z0`, where the periodic simulation box size is `boxsize`.
    """
    for i in range(start_index, end_index):
        x, y, z = pos[i]
        if ((pbc_distance(x, x0, boxsize) < half_width) and (pbc_distance(y, y0, boxsize) < half_width) and (pbc_distance(z, z0, boxsize) < half_width)):  # noqa
            return i

    return None


def find_boxed_from_particles(pos, center, subbox_size, boxsize):
    """
    Find all indices of particles in a box of size `subbox_size` centered on
    `center`, where the simulation box size is `boxsize`.
    """
    if isinstance(center, list):
        center = numpy.asanyarray(center)

    half_width = subbox_size / 2.

    indxs, start_index, end_index = [], 0, len(pos)
    while True:
        i = find_next_particle(start_index, end_index, pos,
                               *center, half_width, boxsize)

        if i is None:
            break

        indxs.append(i)
        start_index = i + 1

    return indxs


def find_boxed(basepath, snap, pkind, fields, center, subbox_size, boxsize,
               comm, verbose=True, dtype=numpy.float32):
    """
    Return particle information of particles in a box of size `subbox_size`
    centered on `center`.

    Parameters
    ----------
    basepath : str
        Path to the simulation output directory.
    snap : int
        Snapshot number.
    pkind : int
        Particle type.
    fields : list of str
        Fields to extract.
    center : list of float
        Center of the sub-box.
    subbox_size : float
        Size of the sub-box.
    boxsize : float
        Size of the periodic simulation box.
    comm : mpi4py.MPI.Comm
        MPI communicator.
    verbose : bool, optional
        Print progress information.
    dtype : type, optional
        Data type to use for the output arrays.

    Returns
    -------
    out : dict of n-dimensional arrays
    """
    if isinstance(fields, str):
        fields = [fields]

    pkind = f"PartType{pkind}"
    files = get_snapshot_files(basepath, snap)

    rank, size = comm.Get_rank(), comm.Get_size()

    # Split the files list into chunks
    chunk_size = len(files) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else len(files)
    nchunks = end - start

    out = {f: [] for f in fields}

    if verbose and rank == 0:
        print(f"Each rank will process ~ {chunk_size} files.", flush=True)

    # Only process the subset of files for this rank
    for n, file in enumerate(files[start:end]):
        if verbose:
            print(f"{datetime.now()}: rank {rank} processed {n}/{nchunks} files. Currently `{file}`.",  # noqa
                  flush=True)

        with File(file, "r") as f:
            pos = f[pkind]["Coordinates"][:].astype(dtype)

            indxs = find_boxed_from_particles(pos, center, subbox_size,
                                              boxsize)

            for field in fields:
                if field == "Coordinates":
                    out[field].append(pos[indxs])
                else:
                    vals = f[pkind][field][:].astype(dtype)
                    out[field].append(vals[indxs])

    # Gather results on the master process
    gathered_data = comm.gather(out, root=0)

    del out
    collect()

    # Combine the results on the master process
    if rank == 0:
        combined_out = {f: [] for f in fields}
        for data in gathered_data:
            for field in fields:
                combined_out[field].extend(data[field])

        for field in fields:
            combined_out[field] = numpy.concatenate(combined_out[field],
                                                    axis=0)

        return combined_out

    return None


def positions_to_density_field(ngrid, pos, center, subbox_size, MAS="PCS",
                               mpart=1., dtype=numpy.float32):
    """
    Convert a set of particle positions to a density field.

    Parameters
    ----------
    ngrid : int
        Number of grid cells per dimension.
    pos : 2-dimensional array of shape (nsamples, 3)
        Particle positions.
    center : 1-dimensional array
        Center of the sub-box.
    subbox_size : float
        Size of the sub-box.
    MAS : str, optional
        Mass assignment scheme.
    mpart : float, optional
        Mass of a single particle.
    dtype : type, optional
        Data type to use for the output array.

    Returns
    -------
    field : 3-dimensional array of shape (ngrid, ngrid, ngrid)
    """
    pos = pos - (center - subbox_size / 2)
    pos = pos.astype(dtype)

    if not numpy.all(pos > 0) and numpy.all(pos < subbox_size):
        raise ValueError("Particles are not within the sub-box.")

    field = numpy.zeros((ngrid, ngrid, ngrid), dtype=dtype)
    MASL.MA(pos, field, subbox_size, MAS, verbose=False)

    print(subbox_size / ngrid)

    field *= mpart / (subbox_size / ngrid)**3

    return field
