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

import numpy as np


import tngsorted

from mpi4py import MPI


basepath = "/mnt/extraspace/rstiskalek/TNG50-1/"
snap = 99
pkind = 1
fields = "Coordinates"
boxsize = 35000.

center = np.array([0., 0., 0.]) + boxsize / 2
subbox_size = 1000.
verbose = True


comm = MPI.COMM_WORLD


out = tngsorted.find_boxed(basepath, snap, pkind, fields, center, subbox_size,
                           boxsize, comm, verbose)

if comm.Get_rank() == 0:
    print("Writing to `test.npz`", flush=True)
    np.savez("test2.npz", **out)
