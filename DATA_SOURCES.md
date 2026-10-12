# Source of the Evaluation Tiles

The evaluation set has 20 image tiles: 10 whole-slide imaging (WSI) tiles in `data/pmd/orig_image/` and 10 serial two-photon tomography (STP) tiles in `data/stp/orig_image/`. Every tile is an unmodified crop of a full-resolution section image from one of the two mouse brains linked on the Brain Architecture Project skeletonization page:
<https://data.brainarchitecture.org/pages/skeletonization>.

This document explains how to recover each tile from those public sources.

## Source brains

| Dataset | Link on the skeletonization page | Viewer series | Brain | Series used |
|---|---|---|---|---|
| WSI | WSI (Visualize) | 850001 | PMD1211 | Fluorescence (`F`) sections |
| STP | STP (Visualize) | 850006 | STP_190322 | STP stitched sections |

## Tile naming convention

Tile file names have the form `<brain>_<section>_<row>_<column>.tif`:

- `<brain>` is `PMD1211` for the WSI tiles and `190322` for the STP tiles.
- `<section>` is the section number in the brain viewer.
- `<row>` and `<column>` give the 1-based pixel position of the tile's top-left corner in the full-resolution section image.

Tiles are 1000 × 1000 pixels, except `190322_59_2501_4431.tif`, which is 900 rows × 1000 columns. WSI tiles have three channels and STP tiles have one; both are 16-bit.

## Tiles available directly from the skeletonization page (4 tiles)

The "Data Samples" section of the page links to two full-resolution section images:

- **WSI Data:** PMD1211 section 58.
- **STP Data:** STP_190322 section 74.

The following four tiles are crops of these two images:

| Tile | Source download |
|---|---|
| `PMD1211_58_5001_12501.tif` | WSI Data |
| `PMD1211_58_8801_13801.tif` | WSI Data |
| `190322_74_2301_5401.tif` | STP Data |
| `190322_74_4001_6501.tif` | STP Data |

## All tiles: downloading through the brain viewer

The full-resolution images of the remaining sections can be downloaded from the brain viewer linked on the page. The table below lists all ten sections, including the two above.

To get a section image:

1. **Look up its file path.** Request the series metadata from the viewer:
   - `https://brainviewer.org:8080/getbraindetails/850001` (WSI)
   - `https://brainviewer.org:8080/getbraindetails/850006` (STP)

   The response lists every section, with its section number (`secnumber`) and image path (`imgfname`). Brain 850001 contains both a Fluorescence and a Nissl series; the tiles come from the Fluorescence series.

2. **Download the image.** Prefix the path with `https://brainviewer.org:8080`. Encode `&` as `%26` in the URL.

| Dataset | Section | Tiles | Image path (`imgfname`) |
|---|---|---|---|
| WSI | 48 | `PMD1211_48_8601_16001.tif`, `PMD1211_48_10701_16601.tif` | `/brainimg/PMD1211/PMD1212&1211-F16-2013.01.11-18.52.14_PMD1211_3_0048.jp2` |
| WSI | 58 | `PMD1211_58_5001_12501.tif`, `PMD1211_58_8801_13801.tif` | `/brainimg/PMD1211/PMD1212&1211-F20-2013.02.04-16.27.36_PMD1211_1_0058.jp2` |
| WSI | 80 | `PMD1211_80_5501_12001.tif`, `PMD1211_80_12501_14501.tif` | `/brainimg/PMD1211/PMD1212&1211-F27-2013.02.04-18.41.58_PMD1211_2_0080.jp2` |
| WSI | 115 | `PMD1211_115_8201_17701.tif`, `PMD1211_115_9501_16001.tif` | `/brainimg/PMD1211/PMD1212&1211-F39-2013.02.04-19.24.12_PMD1211_1_0115.jp2` |
| WSI | 156 | `PMD1211_156_10401_15701.tif`, `PMD1211_156_11301_15501.tif` | `/brainimg/PMD1211/PMD1212&1211-F52-2013.01.12-07.49.51_PMD1211_3_0156.jp2` |
| STP | 47 | `190322_47_2401_4201.tif`, `190322_47_3201_6001.tif` | `/brainimg/STP_190322/StitchedImage_Z047_L001.jp2` |
| STP | 59 | `190322_59_2501_4431.tif`, `190322_59_3301_5601.tif` | `/brainimg/STP_190322/StitchedImage_Z059_L001.jp2` |
| STP | 74 | `190322_74_2301_5401.tif`, `190322_74_4001_6501.tif` | `/brainimg/STP_190322/StitchedImage_Z074_L001.jp2` |
| STP | 80 | `190322_80_2401_5601.tif`, `190322_80_4701_2001.tif` | `/brainimg/STP_190322/StitchedImage_Z080_L001.jp2` |
| STP | 113 | `190322_113_3401_7101.tif`, `190322_113_3701_7801.tif` | `/brainimg/STP_190322/StitchedImage_Z113_L001.jp2` |

For example, the section 48 image is at:

```
https://brainviewer.org:8080/brainimg/PMD1211/PMD1212%261211-F16-2013.01.11-18.52.14_PMD1211_3_0048.jp2
```

## Extracting a tile

Because tile positions are 1-based, subtract 1 from the row and column when indexing the section array:

```python
import glymur  # any JPEG 2000 reader works

section = glymur.Jp2k("PMD1211_1_0058.jp2")[:]
row, col = 5001, 12501            # from PMD1211_58_5001_12501.tif
h, w = 1000, 1000                 # tile height and width
tile = section[row - 1:row - 1 + h, col - 1:col - 1 + w]
```

## Verification

All 20 tiles in `data/pmd/orig_image/` and `data/stp/orig_image/` were extracted this way from the section images listed above. Each extracted crop is identical, pixel for pixel, to the tile in the repository.
