#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include "yet_another_chat/lib/user.h"

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

unsigned char *m_concat(unsigned char *src, unsigned char *concat, size_t size, size_t buffer_size)
{
    src = realloc(src, buffer_size + size + 1);
    memcpy(&src[size], concat, buffer_size);
    return src;
}

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


int main()
{
    FILE *a = fopen("k.png", "rb");
    char header[13] = {0};
    char *buffer;
    struct metadata meta = {0};
    fread(header, 4, 2, a);

    fseek(a, 0, SEEK_END);
    long size = ftell(a);
    fseek(a, 0, SEEK_SET);
    buffer = malloc(size + 1);
    fread(buffer, 1, size, a);
    buffer += 8;
    struct color_alpha *cols = calloc(size, sizeof(struct color_alpha));
    int color_index = 0;
    unsigned char *data = calloc(10, sizeof(char));
    unsigned char *data_ptr = data;
    int color_lenght = 0;
    unsigned long total_lenght = 10;
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
            printf("Image is %ix%i, color type %i, bit depth %i\n", width, height, color_type, bit_depth);
            struct metadata aa = {width, height, bit_depth, color_type, compression, filter, interface};
            meta = aa;
            buffer += 4;
            continue;
            //buffer += 36 + 18;
        }
        else if (startswith(buffer, "IDAT") == 1)
        {
            buffer += 4;
            m_concat(data, (unsigned char *)buffer, total_lenght, lenght);
            total_lenght += lenght;
            buffer += lenght + 4;
            data_ptr += lenght;
            
        }
        else if (startswith(buffer, "IEND"))
        {
            fclose(a);
            break;
        }
        else 
        {
            buffer += lenght + 8;
            printf("advanced %i bytes\n", lenght + 8);
            continue;
        }
    }
    char *uncompressed = calloc(total_lenght, sizeof(char));
    unsigned long src_len = total_lenght;
    uncompress(uncompressed, &total_lenght, data, src_len);
    printf("Uncompressed %i bytes\n", src_len);
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
