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
from os.path import basename

import numpy
import tngsorted
from h5py import File


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
    parser.add_argument("--subboxsize", type=float, default=4000.,
                        help="Sub-box size (in appropriate units matching the particle positions).")                             # noqa
    parser.add_argument("--boxsize", type=float, default=35000.,
                        help="Box size (in appropriate units matching the particle positions).")                                 # noqa
    args = parser.parse_args()

    print(f"{datetime.now()}: loading particle positions.", flush=True)
    with File("/mnt/extraspace/rstiskalek/TNG50-1/output/dmpos_99_downsampled_4.hdf5", "r") as f:  # noqa
        pos = f["pos"][:]
    print(f"{datetime.now()}: done loading particle positions.", flush=True)

    centers = load_centers(args)

    # Create the output file.
    fname_out = f"{basename(args.centers_file).split('.')[0]}.hdf5"
    with File(fname_out, "w") as f:
        f.attrs["subboxsize"] = args.subboxsize
        f.attrs["boxsize"] = args.boxsize
        f.close()

    # Process the centers one by one.
    for i, center in enumerate(centers):
        print(f"{datetime.now()}: processing center {i+1}/{len(centers)}.")

        subpos = tngsorted.find_boxed(pos, center, args.subboxsize,
                                      args.boxsize)

        print(f"{datetime.now()}: writing center {i+1}/{len(centers)}.")
        with File(fname_out, "r+") as f:
            grp = f.create_group(f"center_{i}")
            grp.create_dataset("center", data=center)
            grp.create_dataset("pos", data=subpos)

        print(f"{datetime.now()}: done writing center {i+1}/{len(centers)}.")
