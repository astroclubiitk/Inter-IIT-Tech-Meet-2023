# Image Stitching and Atlas Creation

## Table of Contents

- [Overview](#overview)
- [Files](#files)
    - [Empty_tiles.m](#empty_tilesm)
    - [Populate_tiles.m](#populate_tilesm)
    - [Stitch_tiles.m](#stitch_tilesm)
- [Challenges Faced](#challenges-faced)

## Overview

For image stitching and atlas creation, we opted to use the most popular software used by researchers all around the globe i.e. MATLAB because of it's versitality and default multi-threaded matrix operations which proved invaluable in handing images. Ours is a 3-step approach, with the following files denoting the 3 steps:

## Files

1. empty_tiles.m
2. populate_tiles.m
3. stitch_tiles.m

### Empty_tiles.m

This program creates blank tiles of dimensions 5000px x 5000px which correspond to 2 degree x 2 degree on the lunar surface.

**Naming convention of image**: "\<longitude-180>_\<latitude>.tif"

### Populate_tiles.m

This program divides the TMC images into chunks of 2 degree x 2 degree which then get mapped onto the corresponding tile. The algorithm it uses (loops) is as follows:

- Check whether the respective coordinate difference (both longitude and longitude) between the corners are equal to 0. If not, then skip this file.
- Read the TIF file.
- Rescale the TIF file to half its original resolution.
- Using the coordinate data available from the excel speadsheet break the TIF file into square chunks bounded by the every other longitude and latitude i.e. having a difference of 2 degrees each.
- For each chunk:
    - Read the corresponding tile.
    - For each pixel of the mapping:
        - If pixel value in either the chunk or tile is 0, directly add the two.
        - Else, take the average of both the pixel values.
    - Delete the original tile.
    - Save the new and updated tile in the folder with the same name as the deleted one.
- Take into account all the files that could not be processed.

### Stitch_tiles.m

Due to RAM constraints, we decided to construct the lunar atlas at a resolution of a tenth of the saved tiles. TThe algorithm followed is pretty simple:

- Create a big array that can store the entire lunar atlas.
- For each tile:
    - Read the TIF file.
    - Rescale to required resolution.
    - Insert the image into the large array.
- Save the large array as an image file.

## Challenges Faced

- Time constraint that arised due to download speed and unzipping process, which was addressed by employing distributed computing.
- Not enough RAM. Most commercially available laptops come with a maximum of 16 GB or 32 GB of RAM, which is not enough to even open files with sizes of ~ 50 GB. This issue was addressed by paging the 200 GB laptop memory, essentially making it a virtual RAM, which worked well but slow, compouding the time constraint.