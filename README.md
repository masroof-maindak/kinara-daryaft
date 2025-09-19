# OpenCV CMake Template

Shrimple CMake config to compile C++ OpenCV projects. Assumes a two-directory `src/` & `include/` structure.

## Usage

```bash
git clone https://github.com/<username>/<project-name>.git
cd <project-name>
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. # Or 'Debug' if you want to hack away.
cmake --build .
./demo # args, if relevant
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

<!-- ## Acknowledgements -->
<!---->
<!-- -  -->
