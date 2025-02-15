#include "png_lib.h"

static int startswith(char *buf, char *str)
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

static int decode_int(char **src)
{
    int x= 0;
    memcpy(&x, *src, sizeof(int));
    x = be32toh(x);
    *src += 4;
    return x;
}

static char decode_byte(char **src)
{
    char x= 0;
    memcpy(&x, *src, 1);
    *src += 1;
    return x;
}

static unsigned char decode_ubyte(char **src)
{
    unsigned char x= 0;
    memcpy(&x, *src, 1);
    *src += 1;
    return x;
}

static unsigned char **convert_2d(unsigned char *data, struct metadata meta)
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

static unsigned char *filter_scanline(unsigned char **scanline, int index, int width, int bpp)
{
    unsigned char filter = scanline[index][0];
    printf("Filter for %i is %i\n", index, filter);
    unsigned char *unfiltered = calloc(width * bpp, sizeof(char));
    switch (filter)
    {
        case 0:
            for (int x = 1; x < (width * bpp) + 1;x++)
            {
                unfiltered[x-1] = scanline[index][x];
            }
            break;
        case 1:
            for (int x = 1; x < (width * bpp) + 1;x++)
            {
                int x_trad = x - 1;
                if (x <= bpp)
                    unfiltered[x_trad] = scanline[index][x];
                else
                    unfiltered[x_trad] = scanline[index][x] + unfiltered[x_trad - bpp];
            }
            break;
        case 2:
            for (int x = 1;x < (width * bpp) + 1;x++)
            {
                int x_trad = x - 1;
                if (index == 0)
                    unfiltered[x_trad] = scanline[index][x];
                else
                    unfiltered[x_trad] = scanline[index][x] + scanline[index-1][x_trad];
            }
            break;
        case 3:
            for (int x = 1; x < (width * bpp) + 1; x++)
            {
                int x_trad = x - 1;
                if (index == 0)
                    unfiltered[x_trad] = scanline[index][x];
                else
                {
                    int left = (x > bpp) ? unfiltered[x_trad - bpp] : 0;
                    int up = scanline[index - 1][x_trad];
                    unfiltered[x_trad] = scanline[index][x] + ((left + up) / 2);
                }
            }
            break;
        case 4:
            for (int x = 1; x < (width * bpp) + 1; x++)
            {
                int x_trad = x - 1;
                if (index == 0)
                    unfiltered[x_trad] = scanline[index][x];
                else
                {
                    int left = (x > bpp) ? unfiltered[x_trad - bpp] : 0;
                    int up = scanline[index - 1][x_trad];
                    int up_left = (x > bpp) ? scanline[index - 1][x_trad - bpp] : 0;
                    
                    int p = left + up - up_left;
                    int pa = abs(p - left);
                    int pb = abs(p - up);
                    int pc = abs(p - up_left);
                    
                    if (pa <= pb && pa <= pc)
                        unfiltered[x_trad] = scanline[index][x] + left;
                    else if (pb <= pc)
                        unfiltered[x_trad] = scanline[index][x] + up;
                    else
                        unfiltered[x_trad] = scanline[index][x] + up_left;
                }
            }
            break;
        default:
            for (int x = 1; x < (width * bpp) + 1;x++)
            {
                unfiltered[x-1] = scanline[index][x];
            }
            break;
    }
    return unfiltered;
}

unsigned char **read_png(char *name, struct metadata *meta)
{
    unsigned char magic[9] = {137, 80, 78, 71, 13, 10, 26, 10};
    FILE *a = fopen(name, "rb");
    char header[13] = {0};
    char *buff;
    fread(header, 4, 2, a);
    if (memcmp(header, magic, 8) != 0)
    {
        printf("File is not a png\n");
        return 0;
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
            *meta = aa;
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
    unsigned long dest_len = meta->width * meta->height * 4;
    unsigned char *uncompressed = malloc(dest_len);
    unsigned long src_len = total_lenght;
    uncompress(uncompressed, &dest_len, (unsigned char *)data, total_lenght);
    unsigned char **img = convert_2d(uncompressed, *meta);
    int bpp = 0;
    if (meta->color_type == 2)
        bpp = (3 * meta->bit_depth)/8;
    else if (meta->color_type == 6)
        bpp = (4 * meta->bit_depth)/8;

    for (int y = 0; y < meta->height;y++)
    {
        img[y] = filter_scanline(img, y, meta->width, bpp);
    }
    return img;
}