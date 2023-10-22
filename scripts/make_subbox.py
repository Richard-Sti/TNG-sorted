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
from argparse import ArgumentParser
from os.path import basename
import numpy
import tngsorted
from h5py import File
from mpi4py import MPI
from datetime import datetime


def load_centers(args):
    """Load box centers from a file."""
    centers = numpy.genfromtxt(args.centers_file, delimiter=", ")

    if not (numpy.all(centers > 0) and numpy.all(centers < args.boxsize)):
        raise ValueError("All centers must be within the box.")

    return centers


if __name__ == "__main__":
    parser = ArgumentParser(description="Select particles in a sub-box.")
    parser.add_argument("centers_file", type=str,
                        help="File with box centers. Delimiter expected to be ', ', no header and rows of the form 'x, y, z'.")  # noqa
    parser.add_argument("--basepath", type=str,
                        default="/mnt/extraspace/rstiskalek/TNG50-1/",
                        help="Path to the simulation output directory.")
    parser.add_argument("--snap", type=int, default=99,
                        help="Snapshot number.")
    parser.add_argument("--pkind", type=int, default=1,
                        help="Particle type (1-6).")
    parser.add_argument("--fields", type=str, nargs="+",
                        default=["Coordinates"],
                        help="Particle fields to load.")
    parser.add_argument("--subboxsize", type=float, default=2000.,
                        help="Sub-box size (in appropriate units matching the particle positions).")  # noqa
    parser.add_argument("--boxsize", type=float, default=35000.,
                        help="Box size (in appropriate units matching the particle positions).")  # noqa
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    centers = load_centers(args)
    fname_out = f"{basename(args.centers_file).split('.')[0]}.hdf5"

    # Create output file.
    if rank == 0:
        with File(fname_out, "w") as f:
            f.attrs["basepath"] = args.basepath
            f.attrs["snap"] = args.snap
            f.attrs["pkind"] = args.pkind
            f.attrs["subboxsize"] = args.subboxsize
            f.attrs["boxsize"] = args.boxsize
            f.close()

    # Process the centers one by one.
    for i, center in enumerate(centers):
        if rank == 0:
            print(f"{datetime.now()}: processing center {i+1}/{len(centers)}.")

        comm.Barrier()
        out = tngsorted.find_boxed(args.basepath, args.snap, args.pkind,
                                   args.fields, center, args.subboxsize,
                                   args.boxsize, comm, verbose=False)
        comm.Barrier()

        # Write at rank 0. Create a new group for each center.
        if rank == 0:
            with File(fname_out, "r+") as f:
                grp = f.create_group(f"center_{i}")
                grp.create_dataset("center", data=center)

                for key, val in out.items():
                    grp.create_dataset(key, data=val)

                f.close()
