#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>



int startswith(char *buf, char *str)
{
	int index = 0;
	if (*buf == '\0' && *buf == '\0')
		return 0;
	while (buf[index] != '\0' && str[index] != '\0')
	{
		if (buf[index] != str[index])
			return 0;
		index++;
	}
	return 1;
}

int decode_int(char **src)
{
    int x= 0;
    memcpy(&x, *src, sizeof(int));
    x = be32toh(x);
    *src += 4;
    return x;
}

char decode_byte(char **src)
{
    char x= 0;
    memcpy(&x, *src, 1);
    *src += 1;
    return x;
}

unsigned char decode_ubyte(char **src)
{
    unsigned char x= 0;
    memcpy(&x, *src, 1);
    *src += 1;
    return x;
}

struct raw_pixel
{
    unsigned char *data;
    unsigned char filter;
    int x;
    int y;
};

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

extern struct raw_pixel *wrapper_filter(struct raw_pixel *whole_img, struct raw_pixel *diag_img,int width, int bpp,int height, int filter);

unsigned char **convert_2d(unsigned char *data, struct metadata meta)
{
    unsigned char **ret = (unsigned char **)calloc(meta.height, sizeof(char *));
    int index = 0;
    for (int y = 0; y < meta.height;y++)
    {
        ret[y] = (unsigned char *)calloc((meta.width * 3) + 2, sizeof(char));
        for (int x = 0; x < (meta.width * 3) + 1; x++)
        {
            ret[y][x] = data[index];
            index++;
        }
    }
    return ret;
}

struct raw_pixel **convert_to_raw_pixel(unsigned char **img, struct metadata meta, int bpp)
{
    struct raw_pixel **ret = calloc(meta.height, sizeof(struct raw_pixel *));
    for (int y = 0; y < meta.height;y++)
    {
        ret[y] = calloc(meta.width, sizeof(struct raw_pixel));
        unsigned char filter = img[y][0];
        img[y]++;
        for (int x = 0; x < meta.width; x++)
        {
            struct raw_pixel new;
            new.data = malloc(bpp * sizeof(unsigned char) + 1);
            memcpy(new.data, img[y], bpp);
            new.filter = filter;
            new.x = x;
            new.y = y;
            ret[y][x] = new;
            img[y] += bpp;
        }
    }
    return ret;
}

struct raw_pixel **diagonal_order(struct raw_pixel **img, struct metadata meta)
{
    int y = 0;
    int total_diagonals = meta.height + meta.width - 1;
    struct raw_pixel **ret = calloc(total_diagonals, sizeof(struct raw_pixel *));
    int max_y = meta.height - 1;
    int init_x = 0;
    int img_y = 0;
    
    while (y < total_diagonals)
    {
        ret[y] = calloc(meta.width, sizeof(struct raw_pixel));
        
        int init_y = img_y;
        int img_x = init_x;
        int x = 0;
        
        while (init_y >= 0 && img_x < meta.width)
        {
            ret[y][x] = img[init_y][img_x];
            init_y--;
            img_x++;
            x++;
        }
        
        if (img_y == max_y)
        {
            init_x++;
        }
        else 
        {
            img_y++;
        }
        y++;
    }
    return ret;
}

struct raw_pixel *flatten(struct raw_pixel **img, struct metadata meta)
{
    struct raw_pixel *ret = calloc(meta.width * meta.height + 1, sizeof(struct raw_pixel));
    int index = 0;
    for (int y = 0; y < meta.height;y++)
    {
        for (int x = 0; x < (meta.width * 3) + 1; x++)
        {
            ret[index] = img[y][x];
            index++;
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned char magic[9] = {137, 80, 78, 71, 13, 10, 26, 10};
    FILE *a = fopen("a.png", "rb");
    if (a == NULL)
    {
        printf("File %s doesnt exist!\n", argv[1]);
        return -1;
    }
    char header[13] = {0};
    char *buff;
    struct metadata meta = {0};
    fread(header, 4, 2, a);
    if (memcmp(header, magic, 8) != 0)
    {
        printf("File is not a png\n");
        return -1;
    }

    fseek(a, 0, SEEK_END);
    long size = ftell(a);
    fseek(a, 0, SEEK_SET);
    buff = malloc(size + 1);
    fread(buff, 1, size, a);
    char *buffer = buff;
    unsigned long total_lenght = 0;
    char *data= calloc(size, sizeof(char));
    buffer += 8;
    while (1)
    {
        int lenght = decode_int(&buffer);
        if (startswith(buffer, "IHDR"))
        {
            buffer += 4;
            int width = decode_int(&buffer);
            int height = decode_int(&buffer);
            char bit_depth = decode_byte(&buffer);
            char color_type = decode_byte(&buffer);
            char compression = decode_byte(&buffer);
            char filter = decode_byte(&buffer);
            char interface = decode_byte(&buffer);
            printf("Image is %ix%i, color type %i, bit depth %i filter %i\n", width, height, color_type, bit_depth, filter);
            struct metadata aa = {width, height, bit_depth, color_type, compression, filter, interface};
            meta = aa;
            buffer += 4;
            continue;
            //buffer += 36 + 18;
        }
        else if (startswith(buffer, "IDAT") == 1)
        {
            buffer += 4;
            memcpy(&data[total_lenght], buffer, lenght);
            total_lenght += lenght;
            buffer += lenght + 4;
        }
        else if (startswith(buffer, "IEND"))
        {
            fclose(a);
            break;
        }
        else
        {
            buffer += lenght + 8;
            printf("skipped %i bytes\n", lenght + 8);
            continue;
        }
    }
    total_lenght = total_lenght + 2;
    unsigned long dest_len = meta.width * meta.height * 4;
    unsigned char *uncompressed = malloc(dest_len);
    unsigned long src_len = total_lenght;
    uncompress(uncompressed, &dest_len, (unsigned char *)data, total_lenght);
    unsigned char **raw_img = convert_2d(uncompressed, meta);
    int bpp = 0;
    if (meta.color_type == 2)
        bpp = (3 * meta.bit_depth)/8;
    else if (meta.color_type == 6)
        bpp = (4 * meta.bit_depth)/8;
    struct raw_pixel **img = convert_to_raw_pixel(raw_img, meta,bpp);
    struct raw_pixel **img_diag = diagonal_order(img, meta);
    for (int y = 0; y < meta.height + meta.width - 1;y++)
    {
        for (int x = 0; x < meta.width; x++)
        {
            printf(" Pixel poition %i,%i ", img_diag[y][x].y, img_diag[y][x].x);
        }
        printf("\n");
    }
    struct raw_pixel *img_flat = flatten(img_diag, meta);
    struct raw_pixel *img_diag_flat = flatten(img_diag, meta);
    free(buff);
    free(data);
    free(uncompressed);
    free(raw_img);
    free(img);
    /* 
    for (int y = 0; y < meta.height;y++)
    {
        printf("Filter %i\n", img[y][0]);
        if (y > 0)
            img[y] = wrapper_filter(img[y], img[y-1], meta.width, bpp, img[y][0]); //runs on the gpu yayyy
        else
            img[y] = wrapper_filter(img[y], img[y], meta.width, bpp, img[y][0]); //runs on the gpu yayyy
    }
    int index = 0;
    for (int y = 0; y < meta.height;y++)
    {
        for (int x = 0; x < meta.width; x++)
        {
            unsigned char r = img[y][index];
            unsigned char g = img[y][index + 1];
            unsigned char b = img[y][index + 2];
            if (x == 0)
                printf("\x1b[48;2;%d;%d;%dm ", r, g, b);
            else
            {
                printf(" \x1b[48;2;%d;%d;%dm ", r, g, b);
                //printf(" r: %i g: %i b: %i", r, g, b);
            }
            if (x == meta.width - 1)
            {
                printf("\x1b[48;2;%d;%d;%dm ", r, g, b);
                //printf(" r: %i g: %i b: %i", r, g, b);
            }
            index += 3;
        }
        printf("\x1b[0m\n");
        index = 0;
    }
    free(buff);
    free(data);
    free(uncompressed);
    free(img);*/
    /*for (int i = 1; i <= meta.width*meta.height; i++)
    {
        if (meta.color_type == 2 && meta.bit_depth == 8)
        {
            unsigned char r = decode_ubyte(&data);
            unsigned char g = decode_ubyte(&data);
            unsigned char b = decode_ubyte(&data);

            struct colors col = {r, g, b};
            printf(" %s ", color_string(col, "."));
            if (i%meta.width == 0)
                printf("\n");
        }
        if (meta.color_type == 6 && meta.bit_depth == 8)
        {
            unsigned char r = decode_ubyte(&data);
            unsigned char g = decode_ubyte(&data);
            unsigned char b = decode_ubyte(&data);
            unsigned char a = decode_ubyte(&data);
            struct colors col = {r, g, b};
            printf(" %s ", color_string(col, "."));
            if (i%meta.width == 0)
                printf("\n");
        }
        }*/
    return 0;
}