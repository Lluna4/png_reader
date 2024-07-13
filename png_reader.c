#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lib/user.h"

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

int main()
{
    FILE *a = fopen("aaaa.png", "rb");
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
            printf("Image is %ix%i, color type %i, bit depth %i", width, height, color_type, bit_depth);
            struct metadata aa = {width, height, bit_depth, color_type, compression, filter, interface};
            meta = aa;
            buffer += 36 + 18;
            continue;
        }
        if (startswith(buffer, "IDAT"))
        {
            buffer += 4;
            for (int i = 0;i < lenght;)
            {
                unsigned char r = decode_ubyte(&buffer);
                unsigned char g = decode_ubyte(&buffer);
                unsigned char b = decode_ubyte(&buffer);
                i += 3;
                struct colors col= {r,g,b};
                printf("%s", color_string(col, "."));
                if (i + 3 > lenght)
                {
                    buffer += (4 - (i+3 - lenght) - 1);
                    break;
                }
            }
            buffer += 4;
            continue;
        }
        if (startswith(buffer, "IEND"))
        {
            fclose(a);
            break;
        }
    }
    return 0;
}