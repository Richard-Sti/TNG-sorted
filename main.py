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
Script to load haloes from a TNG simulation and store them in a HDF5 file.
This is motivated by the fact that the default way of accessing haloes'
particles is very slow.

Stores particle coordinates and velocities in a dataset called "particles" and
the offsets of each halo in the dataset in a dataset called "offsets".
"""
from argparse import ArgumentParser
from os.path import join

import h5py
import illustris_python as il
import numpy
from tqdm import trange


def load_halo(hid, basepath):
    data = il.snapshot.loadHalo(basepath, 99, hid, 1,
                                fields=['Coordinates', 'Velocities'])

    out = numpy.zeros((data['count'], 6), dtype=numpy.float32)
    out[:, :3] = data['Coordinates']
    out[:, 3:] = data['Velocities']

    return out


def append_to_h5py(filename, hid, data):
    with h5py.File(filename, 'a') as f:
        parts = f["particles"]

        nparts_before = parts.shape[0]
        nparts_new = data.shape[0]

        parts.resize((nparts_before + nparts_new, parts.shape[1]))
        parts[-nparts_new:, :] = data

        offsets = f["offsets"]
        offsets.resize((offsets.shape[0] + 1, 3))
        offsets[-1, :] = [hid, nparts_before, nparts_before + nparts_new]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--minpart", type=int,
        help="Minimum number of particles in a halo to be processed.")
    parser.add_argument(
        "--basepath", type=str,
        default="/mnt/extraspace/rstiskalek/TNG300-1-Dark/output",
        help="Basepath of the TNG simulation")
    parser.add_argument(
        "--fout_folder", type=str,
        default="/mnt/extraspace/rstiskalek/TNG300-1-Dark",
        help="Folder where the output file will be stored")
    args = parser.parse_args()

    lengths = il.groupcat.loadHalos(args.basepath, 99, fields="GroupLen")
    mask = lengths > args.minpart

    ntot = numpy.sum(mask)
    nparts = numpy.sum(lengths[mask])
    size = nparts * 6 * 4 / 1024**3
    size = float("%.4g" % size)
    est_time = ntot * 150 / 1000 / 60**2
    est_time = float("%.4g" % est_time)

    print(f"Number of halos to be processed:   {ntot}.")
    print(f"Total number of particles:         {nparts}.")
    print(f"Estimated size of the output file: {size} GB.")
    print(f"Estimated time to process:         {est_time} hours.")

    fout = join(args.fout_folder, "sorted_halos.hdf5")
    f = h5py.File(fout, 'w')
    dset = f.create_dataset("particles", shape=(0, 6), maxshape=(None, 6),
                            dtype=numpy.float32)
    dset = f.create_dataset("offsets", shape=(0, 3), maxshape=(None, 3),
                            dtype=numpy.int64)
    f.close()

    for hid in trange(ntot, mininterval=5):
        halo = load_halo(hid, args.basepath)
        append_to_h5py(fout, hid, halo)
