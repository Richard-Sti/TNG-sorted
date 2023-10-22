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
from glob import glob
from os.path import join


def get_snapshot_files(basepath, snap):
    """
    List all files for a given snapshot.

    Parameters
    ----------
    basepath : str
        Path to the simulation output directory.
    snap : int
        Snapshot number.

    Returns
    -------
    files : list of str
    """
    snap = str(snap).zfill(3)
    fpath = join(basepath, "output", f"snapdir_{snap}", f"snap_{snap}*.hdf5")
    return glob(fpath)


from .select_box import find_boxed, positions_to_density_field                  # noqa
