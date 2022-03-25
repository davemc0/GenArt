#pragma once

// How many samples in the Halton sequence
#define NUM_SAMPLE_LOCS 128

// Total size of token array, shared across all three channels
#define MAX_TOKENS 4096

const int MAX_COLORMAP_ENTRIES = 64;

// The dimensions of the thread block for CUDA
// Make the individual warps more square.
// A warp is 8x4, so a block is 1x8 warps.
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 32

#define CMAP_SIZE 4

enum ColorMapOrderings_t { REORDER_SHORT = 0, REORDER_LONG = 1, REORDER_RANDOM = 2, REORDER_LUM = 3, NUM_REORDERS = 4 };

static const char* ColorMapOrderings[NUM_REORDERS] = {"low contrast", "high contrast", "random", "luminance"};

// The various color spaces
enum ColorSpace_t {
    SPACE_RGB = 0,
    SPACE_TONEMAP_RGB = 1,
    SPACE_YCRCB = 2,
    SPACE_TONEMAP_HSV = 3,
    SPACE_COLMAP = 4,
    SPACE_TONEMAP_COLMAP = 5,
    NUM_COLORSPACES = 6
};

static const char* ColorSpaceNames[NUM_COLORSPACES] = {"RGB", "Toned RGB", "YCrCb", "Toned HSV", "ColorMap", "Toned ColorMap"};
