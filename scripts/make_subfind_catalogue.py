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
Script to combine the subfind catalogue and supplementary catalogues.
"""
from argparse import ArgumentParser
from os.path import join

import h5py
import illustris_python as il
import numpy as np


def load_subfind(basepath, snap, fields):
    """
    Load subfind catalogue and process it.
    """
    path = join(basepath, "output")
    if "SubhaloFlag" not in fields:
        fields = ["SubhaloFlag"] + fields
    return il.groupcat.loadSubhalos(path, snap, fields=fields)


def append_neutral_hydrogen(data, basepath, snap):
    """
    Append neutral hydrogen mass to the subfind catalogue.
    """
    path = join(basepath, "postprocessing",
                "hih2_galaxy", f"hih2_galaxy_{str(snap).zfill(3)}.hdf5")
    with h5py.File(path, "r") as f:
        subhalo_id = f["id_subhalo"][:].astype(int)

    keys_extract = ["m_neutral_H"]

    with h5py.File(path, "r") as f:
        for key in keys_extract:
            _x = f[key][:]

            x = np.full(data["count"], np.nan, dtype=_x.dtype)
            for i, j in enumerate(subhalo_id):
                x[j] = _x[i]

            data[key] = x

    return data


def write_dict_to_hdf5(data, basepath, snap):
    """
    Write a dictionary to an HDF5 file.
    """
    fname = join(basepath,
                 "postprocessing",
                 f"catalogue_{str(snap).zfill(3)}.hdf5"
                 )

    print(f"Writing to `{fname}`.")
    with h5py.File(fname, "w") as f:
        for key, val in data.items():
            if key == "count":
                continue
            f.create_dataset(key, data=val)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--basepath", type=str,
                        default="/mnt/extraspace/rstiskalek/TNG50-1/",
                        help="Path to the simulation directory.")
    parser.add_argument("--snap", type=int, default=99,
                        help="Snapshot number.")
    args = parser.parse_args()

    subfind_fields = ["SubhaloMass", "SubhaloCM", "SubhaloMassType"]

    data = load_subfind(args.basepath, args.snap, subfind_fields)
    data = append_neutral_hydrogen(data, args.basepath, args.snap)

    write_dict_to_hdf5(data, args.basepath, args.snap)
