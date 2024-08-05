#ifndef PNG_LIB_H
# define PNG_LIB_H

#include <string.h>
#include <zlib.h>
#include <endian.h>
#include <stdlib.h>
#include <stdio.h>

struct metadata
{
    int width;
    int height;
    char bit_depth;
    char color_type;
    char compression;
    char filter;
    char interface;
};

struct color_alpha
{
    int r;
    int g;
    int b;
    int alpha;
};

struct colors
{
	int r, g, b;
};

unsigned char **read_png(char *name, struct metadata *meta);

#endif