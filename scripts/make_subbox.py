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
from argparse import ArgumentParser
from datetime import datetime
from os.path import join

import numpy
import tngsorted
from h5py import File
from mpi4py import MPI


def load_centers(centers_file, boxsize):
    """Load box centers from a file."""
    data = numpy.genfromtxt(centers_file, delimiter=", ", skip_header=1)

    # Shuffle so that the zeroth rank does not have the most massive haloes.
    gen = numpy.random.default_rng(seed=42)
    gen.shuffle(data)

    ids = data[:, 0].astype(int)
    centers = data[:, 1:4]

    if not (numpy.all(centers > 0) and numpy.all(centers < boxsize)):
        raise ValueError("All centers must be within the box.")

    return ids, centers.reshape(-1, 3)


if __name__ == "__main__":
    parser = ArgumentParser(description="Make density fields around haloes.")
    parser.add_argument("centers_file", type=str,
                        help="File with box centers. Delimiter expected to be ', ', no header and rows of the form 'x, y, z'.")  # noqa
    args = parser.parse_args()

    pospath = "/mnt/extraspace/rstiskalek/TNG50-1/output/dmpos_99_downsampled_4.hdf5"  # noqa
    # pospath = "/mnt/extraspace/rstiskalek/TNG50-1/output/dm_pos_128.hdf5"
    dumpfolder = "/mnt/extraspace/rstiskalek/TNG50-1/postprocessing/density_field"  # noqa
    rate = 4
    boxsize = 35000.
    subbox_size = 2000.
    mpart = 3.07367708626464e-05 * 1e10 * rate
    ngrid = 128
    MAS = "PCS"

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        print(f"{datetime.now()}: loading particle positions.", flush=True)
    with File(pospath, 'r') as f:
        pos = f["pos"][:]

    comm.Barrier()
    if rank == 0:
        print(f"{datetime.now()}: all ranks loaded particle positions.",
              flush=True)

    ids, centers = load_centers(args.centers_file, boxsize)

    # Split centers and ids into chunks for each rank.
    chunk_size = len(centers) // size
    extra_tasks = len(centers) % size
    start_idx = rank * chunk_size + min(rank, extra_tasks)
    end_idx = start_idx + chunk_size + (1 if rank < extra_tasks else 0)

    centers = centers[start_idx:end_idx]
    ids = ids[start_idx:end_idx]

    for i in range(len(centers)):
        center = centers[i]
        print(f"Rank {rank}, {datetime.now()}: processing center {i+1}/{len(centers)}.", flush=True)  # noqa
        fname_out = join(dumpfolder, f"subhalo_{ids[i]}.npz")

        subpos = tngsorted.find_boxed(pos, center, subbox_size, boxsize)

        field = tngsorted.positions_to_density_field(
            ngrid, subpos, center, subbox_size, boxsize, mpart=mpart,
            MAS=MAS, verbose=False)

        numpy.savez(fname_out, field=field, center=center,
                    subbox_size=subbox_size, ngrid=ngrid, MAS=MAS)
