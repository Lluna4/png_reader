#include <string.h>
#include <wayland-client-core.h>
#include <wayland-client-protocol.h>
#include <wayland-client.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "xdg-shell-client-protocol.h"
#include <string.h>
#include <zlib.h>
#include <endian.h>


struct wl_compositor *compositor;
struct wl_surface *surface;
struct wl_buffer *frame_buff;
struct wl_shm *shared_memory;
struct xdg_wm_base *shell;
struct xdg_toplevel *toplevel;

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

struct color_alpha
{
    int r;
    int g;
    int b;
    int alpha;
};

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

unsigned char *filter_scanline(unsigned char **scanline, int index, int width, int bpp)
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


unsigned char *pixel;
int width = 300;
int height = 300;
uint8_t c = 0;
uint8_t cl = 0;
unsigned char **img;
struct metadata meta;

int32_t allocate_shared_memory(uint64_t size)
{
    int8_t name[8];
    name[0] = '/';
    name[7] = 0;
    for (int i = 1; i < 6; i++)
        name[i] = 'a' + i;
    
    int32_t fd = shm_open(name, O_RDWR | O_EXCL | O_CREAT, S_IWUSR | S_IRUSR | S_IWOTH | S_IROTH);
    shm_unlink(name);
    ftruncate(fd, size);
    
    return fd;
}

void resize()
{
    int32_t fd = allocate_shared_memory(width * height * 4);
    
    pixel = mmap(0, width * height * 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    struct wl_shm_pool *pool = wl_shm_create_pool(shared_memory, fd, width * height * 4);
    frame_buff = wl_shm_pool_create_buffer(pool, 0, width, height, width * 4, WL_SHM_FORMAT_ABGR8888);
    wl_shm_pool_destroy(pool);
    close(fd);
}

void draw()
{
    int index = 0;
    int pxl_index = 0;
    for (int y = 0; y < meta.height;y++)
    {
        for (int x = 0; x < meta.width; x++)
        {
            unsigned char r = img[y][index];
            unsigned char g = img[y][index + 1];
            unsigned char b = img[y][index + 2];
            pixel[pxl_index] = r;
            pxl_index++;
            pixel[pxl_index] = g;
            pxl_index++;
            pixel[pxl_index] = b;
            pxl_index++;
            pixel[pxl_index] = 255;
            pxl_index++;
            index += 3;
        }
        index = 0;
        pxl_index += (width * 4) - (meta.width * 4);
    }
    wl_surface_attach(surface, frame_buff, 0, 0);
    wl_surface_damage_buffer(surface, 0, 0, width, height);
    wl_surface_commit(surface);
}

struct wl_callback_listener callback_listener;

void render_frame(void *data, struct wl_callback *callback, unsigned int callback_data)
{
    wl_callback_destroy(callback);
    callback = wl_surface_frame(surface);
    wl_callback_add_listener(callback, &callback_listener, 0);
    c++;
    draw();
}

struct wl_callback_listener callback_listener = {.done = render_frame};

void shell_ping(void *data, struct xdg_wm_base *sh, unsigned int serial)
{
    xdg_wm_base_pong(sh, serial);
}

void toplevel_configure(void *data, struct xdg_toplevel *toplevel, int nwidth, int nheight, struct wl_array *states)
{
    if (!nwidth && !nheight)
        return;
    
    if (height != nheight || width != nwidth)
    {
        munmap(pixel, height * width * 4);
        height = nheight;
        width = nwidth;
        resize();
    }
}
void toplevel_close(void *data,struct xdg_toplevel *xdg_toplevel)
{
    cl = 1;
}

void toplevel_configure_bounds(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height)
{
    
}

void toplevel_wm_capabilities(void *data, struct xdg_toplevel *xdg_toplevel, struct wl_array *capabilities)
{
    
}

void surface_xdg_configure(void *data, struct xdg_surface *surface_xdg, unsigned int serial)
{
    xdg_surface_ack_configure(surface_xdg, serial);
    if (!pixel)
        resize();
    draw();
}

struct xdg_surface_listener surface_xdg_listener = {surface_xdg_configure};
struct xdg_toplevel_listener toplevel_listener = {.configure = toplevel_configure, .close = toplevel_close, .configure_bounds = toplevel_configure_bounds,
    .wm_capabilities = toplevel_wm_capabilities};
struct xdg_wm_base_listener shell_listener = {.ping = shell_ping};

void reg_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version)
{
    if (strcmp(interface, wl_compositor_interface.name) == 0)
    {
        compositor = wl_registry_bind(wl_registry, name, &wl_compositor_interface, 4);
        printf("Compositor binded!\n");
    }
    else if (strcmp(interface, wl_shm_interface.name) == 0)
    {
        shared_memory = wl_registry_bind(wl_registry, name, &wl_shm_interface, 1);
    }
    else if (strcmp(interface, xdg_wm_base_interface.name) == 0)
    {
        shell = wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, 1);
        xdg_wm_base_add_listener(shell, &shell_listener, 0);
    }
}
void reg_global_remove(void *data, struct wl_registry *wl_registry, uint32_t nam)
{
    
}
struct wl_registry_listener listener = {.global = reg_global, .global_remove = reg_global_remove};

unsigned char **read_png(char *name)
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
    unsigned char **img = convert_2d(uncompressed, meta);
    int bpp = 0;
    if (meta.color_type == 2)
        bpp = (3 * meta.bit_depth)/8;
    else if (meta.color_type == 6)
        bpp = (4 * meta.bit_depth)/8;

    for (int y = 0; y < meta.height;y++)
    {
        img[y] = filter_scanline(img, y, meta.width, bpp);
    }
    return img;
}


int main(int argc, char *argv[])
{
    struct wl_display *disp = wl_display_connect(0);
    if (argc < 2)
    {
        printf("Not enough arguments!\n");
        return -1;
    }
    if (!disp)
        return -1;
    printf("Display connected!\n");
    
    struct wl_registry *reg = wl_display_get_registry(disp);
    wl_registry_add_listener(reg, &listener, 0);
    wl_display_roundtrip(disp);
    surface = wl_compositor_create_surface(compositor);
    struct wl_callback *callback = wl_surface_frame(surface);
    wl_callback_add_listener(callback, &callback_listener, 0);
    
    struct xdg_surface *surface_xdg = xdg_wm_base_get_xdg_surface(shell, surface);
    xdg_surface_add_listener(surface_xdg, &surface_xdg_listener, 0);
    toplevel = xdg_surface_get_toplevel(surface_xdg);
    xdg_toplevel_add_listener(toplevel, &toplevel_listener, 0);
    xdg_toplevel_set_title(toplevel, "AAAAAAAAAAAA");
    wl_surface_commit(surface);
    
    img = read_png(argv[1]);
    if (!img)
        return -1;
    
    while (wl_display_dispatch(disp) && cl == 0);
    
    if (frame_buff)
    {
        wl_buffer_destroy(frame_buff);
    }
    xdg_toplevel_destroy(toplevel);
    xdg_surface_destroy(surface_xdg);
    wl_surface_destroy(surface);
    wl_display_disconnect(disp);
    return 0;
}