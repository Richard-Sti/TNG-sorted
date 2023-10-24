# Copyright (C) 2023 Richard Stiskalek
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
from glob import glob
from os import remove
from os.path import join

import numpy
from h5py import File
from tqdm import tqdm

if __name__ == "__main__":
    dumpfolder = "/mnt/extraspace/rstiskalek/TNG50-1/postprocessing/density_field"  # noqa
    rate = 4
    ngrid = 128
    MAS = "PCS"

    files = glob(join(dumpfolder, "subhalo_*.npz"))
    hids = numpy.sort([int(file.split("_")[-1].split(".")[0])
                       for file in files])

    fname_out = join(dumpfolder,
                     f"fields_rate_{rate}_ngrid_{ngrid}_MAS_{MAS}.hdf5")

    with File(fname_out, "w") as f:
        for hid in tqdm(hids, desc="Writing sub-boxes"):
            fname = join(dumpfolder, f"subhalo_{hid}.npz")
            x = numpy.load(fname, allow_pickle=True)
            grp = f.create_group(str(hid))
            grp.create_dataset("field", data=x["field"])
            grp.create_dataset("center", data=x["center"])
            grp.create_dataset("subbox_size", data=[x["subbox_size"]])
            grp.create_dataset("ngrid", data=[x["ngrid"]])

            remove(fname)

        f.close()
