# NeuTube Build & Setup Guide

This document describes all the modifications required to build and run **neuTube** on a modern Linux system with Python 3 and current compilers (GCC 11+).

## Overview

neuTube is a neuron tracing and skeletonization software. The original codebase was designed for Python 2 and older compilers, requiring several patches to work with modern systems.

## Prerequisites

- Linux (tested on Ubuntu 22.04+)
- Python 3.10+
- GCC 11+
- SWIG (for Python bindings)
- CMake 3.x
- Basic development tools: make, autoconf, etc.

## Build Process

### 1. Build External Libraries

```bash
cd neutube/neurolabi/lib
./build.sh
```

**Modifications made to `neurolabi/lib/build.sh`:**
- Changed `use_hdf5=0` to `use_hdf5=1` (line 113) to enable HDF5 build
- Changed all `make` commands to `make -j8` for parallel compilation (lines 43, 47, 65, 83, 104, 126)

### 2. Build C Library

```bash
cd neutube/neurolabi
./update_library
```

**Modifications made to `neurolabi/update_library`:**
- Changed `make $lib_config AFLAGS="$AFLAGS"` to `make -j8 $lib_config AFLAGS="$AFLAGS"` (line 92)

### 3. Build C++ Library

```bash
cd neutube/neurolabi/cpp/lib
./build.sh
```

**Modifications made to `neurolabi/cpp/lib/build.sh`:**
- Changed `make` to `make -j8` (line 12)

### 4. Build Python Module

```bash
cd neutube/neurolabi/python/module
make
```

**Modifications made to `neurolabi/python/module/Makefile`:**
- Changed `python-config` to `python3-config` (lines 12, 15)
- Added include paths for FFTW, PNG, XML, and HDF5:
  ```
  -I../../lib/fftw3/include -I../../lib/png/include -I../../lib/xml/include/libxml2 -I../../lib/hdf5/include
  ```

## C++ Code Modifications

### File: `neurolabi/gui/zqtheader.h`

Added dummy methods to `ZQColor` class for non-Qt builds:

```cpp
// Before (line 43-44):
class ZQColor{};
#define QColor ZQColor

// After:
class ZQColor{
public:
  int red() const { return 0; }
  int green() const { return 0; }
  int blue() const { return 0; }
};
#define QColor ZQColor
```

### File: `neurolabi/gui/zstack.hxx`

1. Commented out ambiguous constructor declaration (lines 74-75):
```cpp
// Before:
  ZStack(Mc_Stack *stack/*,
         C_Stack::Mc_Stack_Deallocator *dealloc = C_Stack::kill*/);

// After:
//  ZStack(Mc_Stack *stack/*,
//         C_Stack::Mc_Stack_Deallocator *dealloc = C_Stack::kill*/);
```

2. Added `#include <QString>` within `_NEUTUBE_` guard block (lines 11-16)

3. Wrapped `std::vector<QString> m_lsmChannelNames;` with `#ifdef _NEUTUBE_` (lines 573-579)

### File: `neurolabi/gui/zstack.cxx`

Commented out the corresponding constructor definition (lines 83-89):
```cpp
// Before:
ZStack::ZStack(Mc_Stack *stack/*, C_Stack::Mc_Stack_Deallocator *dealloc*/) :
  m_stack(NULL)
{
  init();
  setData(stack, C_Stack::kill);
}

// After:
//ZStack::ZStack(Mc_Stack *stack/*, C_Stack::Mc_Stack_Deallocator *dealloc*/) :
//  m_stack(NULL)
//{
//  init();
//  setData(stack, C_Stack::kill);
//}
```

### File: `neurolabi/gui/zstackprocessor.cpp`

Added missing include (line 1):
```cpp
#include <cstring>  // for memcpy
```

## C Code Modifications (GCC 14+ Compatibility)

Modern GCC (14+) treats implicit function declarations as errors. The following C files needed `_GNU_SOURCE` or `<strings.h>` to expose BSD/GNU extensions.

### File: `neurolabi/c/tz_image_lib.c`

Changed conditional include to unconditional (lines 4-6):
```c
// Before:
#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif

// After:
#include <strings.h>
```
This fixes the `bzero` implicit declaration error.

### File: `neurolabi/c/tz_pipe.c`

Added `_GNU_SOURCE` at the very first line (before any includes):
```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```
This exposes the `strsep` function.

### File: `neurolabi/c/tz_image_io.c`

Added `_GNU_SOURCE` at line 6 (before standard library includes):
```c
#define _GNU_SOURCE
#include "tz_image_io.h"
#include <stdio.h>
...
```
This exposes `fileno`, `ftello`, `fseeko`, and `strsep`.

### File: `neurolabi/c/tz_swc_tree.c`

Added `_GNU_SOURCE` at the very first line:
```c
#define _GNU_SOURCE
/* @file tz_swc_tree.c ... */
```
This exposes the `strsep` function.

### File: `neurolabi/lib/genelib/src/cdf.p`

Added `_DEFAULT_SOURCE` and `_XOPEN_SOURCE` before includes (after the header comment, before line 12):
```c
/* Enable POSIX functions like drand48() when compiling with -std=c99 */
#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE

#include <stdio.h>
```
This fixes the `drand48` implicit declaration error when compiling with `-std=c99`.

### File: `neurolabi/lib/genelib/src/utilities.p`

Added `_DEFAULT_SOURCE` and `_XOPEN_SOURCE` before includes (after the license header):
```c
/* Enable POSIX functions like strdup() when compiling with -std=c99 */
#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE

#include <stdlib.h>
```
This fixes the `strdup` implicit declaration error.

### File: `neurolabi/lib/genelib/src/water_shed.p`

Added `_DEFAULT_SOURCE`, `_XOPEN_SOURCE`, and `<strings.h>` include before other includes:
```c
/* Enable POSIX functions like bzero() when compiling with -std=c99 */
#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE
#include <strings.h>

#include <stdio.h>
```
This fixes the `bzero` implicit declaration error.

### File: `neurolabi/lib/genelib/src/tiff_io.p`

Added `_DEFAULT_SOURCE` and `_XOPEN_SOURCE 500` before includes (after the header comment):
```c
/* Enable POSIX functions like fileno(), ftruncate(), mkstemp() when compiling with -std=c99 */
#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 500

#include <stdlib.h>
```
This fixes the `fileno`, `ftruncate`, and `mkstemp` implicit declaration errors.

## CMake Compatibility (CMake 3.31+)

### File: `neurolabi/cpp/lib/CMakeLists.txt`

Updated minimum version (line 1):
```cmake
// Before:
cmake_minimum_required(VERSION 2.8)

// After:
cmake_minimum_required(VERSION 3.5)
```

## Python Code Modifications

### File: `neurolabi/python/QMakeParser.py`

Updated print statements for Python 3 (lines 48, 71-73):
```python
# Before:
print line

# After:
print(line)
```

### File: `neurolabi/python/skeletonize.py`

Multiple Python 3 compatibility changes:
1. Changed `import httplib` to `import http.client as httplib` (line 11)
2. Updated all `print` statements to use parentheses
3. Changed `config.has_key('x')` to `'x' in config` (all occurrences)
4. Changed `config['args'].has_key('x')` to `'x' in config['args']` (all occurrences)

### File: `neurolabi/python/flyem/LoadDvidObject.py`

1. Changed `import httplib` to `import http.client as httplib` (line 8)
2. Updated all `print` statements to use parentheses

## Configuration

### Skeletonization Config (`neurolabi/json/skeletonize.json`)

```json
{
  "downsampleInterval": [1, 1, 1],
  "minimalLength": 40,
  "keepingSingleObject": true,
  "rebase": true,
  "fillingHole": true,
  "maximalDistance": 100
}
```

**Note:** `downsampleInterval: [1, 1, 1]` enables 2x downsampling for faster processing. Use `[0, 0, 0]` for full resolution (slower but more accurate).

## Usage

### Running Skeletonization

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/neutube/neurolabi/python/module

python3 /path/to/neutube/neurolabi/python/skeletonize.py \
  -i input_image.tif \
  -o output_skeleton.swc \
  --config /path/to/neutube/neurolabi/json/skeletonize.json
```

### Batch Processing

See `batch_neutube.py` in the tilesPMD directory for batch processing multiple images.

### Converting SWC to TIF

See `swc_to_tif.py` in the tilesPMD directory for converting SWC skeleton files to TIF images.

## Evaluation Results

Using `evaluate_model.py` with distance threshold of 5 pixels:

| Metric | Value |
|--------|-------|
| Precision | 29.22% |
| Recall | 35.16% |
| F-Score | 31.92% |
| Dice | 31.92% |
| IoU | 18.99% |

## Troubleshooting

### Segmentation Fault
If you encounter segfaults, try:
1. Using a smaller image first to verify the build
2. Checking if all libraries are properly linked

### Missing Headers
If you see errors about missing headers (e.g., `fftw3.h`), ensure the library build completed successfully and the include paths are correct in the Makefile.

### Python Import Errors
Ensure `PYTHONPATH` includes the module directory:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/neutube/neurolabi/python/module
```

## Files Modified Summary

| File | Changes |
|------|---------|
| `neurolabi/lib/build.sh` | Enable HDF5, parallel make |
| `neurolabi/update_library` | Parallel make |
| `neurolabi/cpp/lib/build.sh` | Parallel make |
| `neurolabi/cpp/lib/CMakeLists.txt` | CMake 3.5+ minimum version |
| `neurolabi/python/module/Makefile` | Python 3, include paths |
| `neurolabi/gui/zqtheader.h` | ZQColor dummy methods |
| `neurolabi/gui/zstack.hxx` | Constructor fix, QString include |
| `neurolabi/gui/zstack.cxx` | Constructor fix |
| `neurolabi/gui/zstackprocessor.cpp` | Include cstring |
| `neurolabi/c/tz_image_lib.c` | Unconditional strings.h include |
| `neurolabi/c/tz_pipe.c` | _GNU_SOURCE for strsep |
| `neurolabi/c/tz_image_io.c` | _GNU_SOURCE for fileno/ftello/fseeko |
| `neurolabi/c/tz_swc_tree.c` | _GNU_SOURCE for strsep |
| `neurolabi/lib/genelib/src/cdf.p` | _DEFAULT_SOURCE/_XOPEN_SOURCE for drand48 |
| `neurolabi/lib/genelib/src/utilities.p` | _DEFAULT_SOURCE/_XOPEN_SOURCE for strdup |
| `neurolabi/lib/genelib/src/water_shed.p` | _DEFAULT_SOURCE/_XOPEN_SOURCE/strings.h for bzero |
| `neurolabi/lib/genelib/src/tiff_io.p` | _DEFAULT_SOURCE/_XOPEN_SOURCE for fileno/ftruncate/mkstemp |
| `neurolabi/python/QMakeParser.py` | Python 3 print |
| `neurolabi/python/skeletonize.py` | Python 3 compatibility |
| `neurolabi/python/flyem/LoadDvidObject.py` | Python 3 compatibility |
| `neurolabi/json/skeletonize.json` | Downsampling config |
