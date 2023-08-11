# TNG-sorted

This repository contains a script to efficiently load haloes from a TNG simulation and store them in an HDF5 file.

## Motivation

The primary motivation for this script is that the default method of accessing haloes' particles can be quite slow. This utility provides a more optimized approach by storing particle coordinates and velocities in an HDF5 dataset named `particles`, and the offsets of each halo in the dataset named `halomap`.

## Dependencies

- `h5py`
- `illustris_python`
- `numpy`
- `tqdm`

## Usage

To run the script, execute:

```
python main.py --minpart MIN_NUMBER_OF_PARTICLES --basepath BASE_PATH_OF_TNG_SIMULATION --fout_folder OUTPUT_FOLDER_PATH
```

### Arguments:

- `--minpart`: Minimum number of particles in a halo to be processed.
- `--basepath`: Base path of the TNG simulation.
- `--fout_folder`: Folder where the output file will be stored.

## Output

The script will generate an HDF5 file named `sorted_halos.hdf5` in the specified output folder. This file will contain:
- A dataset named `particles` that holds particle coordinates and velocities.
- A dataset named `offsets` that stores the offsets of each halo in the `particles` dataset.

## Author

- Richard Stiskalek
