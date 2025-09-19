# Kinara Daryaft

Despicable implementation of a Canny Edge Detector in C++.

## Usage

```bash
git clone https://github.com/masroof-maindak/kinara-daryaft.git
cd kinara-daryaft
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. # Or 'Debug' if you want to hack away.
cmake --build .
./knr -i <input-image> -o <output-dir># args, if relevant
```

## Dependencies

- OpenCV
- QT6
- VTK
- HDF5

#### Arch Linux

```bash
sudo pacman -S opencv qt6-base vtk hdf5
```

## Acknowledgements

- Fundamentals of Computer Vision - Mubarak Shah
- [ArgParse](https://github.com/p-ranav/argparse)
