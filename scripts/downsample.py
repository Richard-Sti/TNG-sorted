import numpy as np
from datetime import datetime
import h5py
import os
from tqdm import tqdm
from argparse import ArgumentParser

from os.path import join

import illustris_python as il


def downsample_particles(dm_pos, rate):
    """Note that `dm_pos` is shuffled in-place."""
    print(f"{datetime.now()}:   shuffling particles.", flush=True)
    np.random.shuffle(dm_pos)
    print(f"{datetime.now()}:   finished shuffling particles.", flush=True)

    return dm_pos[::rate]


def check_hdf5_files(directory):
    """Check if all HDF5 files in a directory are readable."""
    corrupted_files = []

    files = [f for f in os.listdir(directory) if f.endswith('.hdf5')]
    for file in tqdm(files, desc="Checking HDF5 files"):
        filepath = os.path.join(directory, file)
        try:
            # Try opening the file with h5py
            with h5py.File(filepath, 'r'):
                pass
        except Exception as e:  # noqa
            # If there's an error, add the file to the corrupted list
            corrupted_files.append(filepath)

    if corrupted_files:
        print("The following files appear to be corrupted:")
        for f in corrupted_files:
            print(f)
    else:
        print("All HDF5 files in the directory seem to be fine.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rate", type=int, required=True,
                        help="Downsampling rate.")
    parser.add_argument("--basepath", type=str,
                        default="/mnt/extraspace/rstiskalek/TNG50-1/output",
                        help="Basepath to the simulation output.")
    parser.add_argument("--nsnap", type=int, default=99,
                        help="Snapshot number.")
    parser.add_argument("--check_corrupt", action="store_true",
                        help="Check for corrupt HDF5 files in the directory.")
    args = parser.parse_args()

    if args.check_corrupt:
        fpath = join(args.basepath, f"snapdir_{args.nsnap}")
        check_hdf5_files(fpath)
        quit()

    if args.rate < 0:
        raise ValueError("The downsampling rate must be non-negative.")

    print(f"{datetime.now()}: loading the DM particle positions.", flush=True)
    pos = il.snapshot.loadSubset(args.basepath, args.nsnap, "dm",
                                 fields=["Coordinates"], float32=True)
    print(f"{datetime.now()}: finished loading the DM particle positions.",
          flush=True)

    print(f"{datetime.now()}: downsampling the DM particle positions.",
          flush=True)
    if args.rate > 0:
        pos = downsample_particles(pos, args.rate)

    fout = join(args.basepath,
                f"dmpos_{args.nsnap}_downsampled_{args.rate}.hdf5")
    with h5py.File(fout, 'w') as f:
        f.create_dataset("pos", data=pos)

    print(f"{datetime.now()}: wrote the DM particle positions to {fout}.",
          flush=True)
