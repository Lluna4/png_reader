#include <stdlib.h>
#include <stdio.h>

struct raw_pixel
{
    unsigned char *data;
    unsigned char filter;
    int x;
    int y;
};

__global__ void filter_scanline(struct raw_pixel **scanline, struct raw_pixel **prev_scanline, struct raw_pixel **unfiltered, int width, int bpp, int filter)
{
    int x  = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int x_trad = x;
    int pos_x = scanline[y][x].x;
    int pos_y = scanline[y][x].y;
    switch (filter)
    {
        case 0:
                unfiltered[pos_y][pos_x] = scanline[y][x];
            break;
        case 1:
            if (pos_x < 1)
                unfiltered[pos_y][pos_x] = scanline[y][x];
            else
                for (int i = 0; i < bpp; i++)
                {
                    unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + unfiltered[pos_y][pos_x - 1].data[i];
                }
                
            break;
        case 2:
            if (pos_y < 1)
                unfiltered[pos_y][pos_x] = scanline[y][x];
            else
                for (int i = 0; i < bpp; i++)
                {
                    unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + prev_scanline[pos_y -1][pos_x].data[i];
                }
            break;
        case 3:
            if (pos_y < 1)
                unfiltered[pos_y][pos_x] = scanline[y][x];
            else
            {
                raw_pixel left = (pos_x > 0) ? unfiltered[pos_y][pos_x -1] : (raw_pixel){0};
                raw_pixel up = prev_scanline[pos_y - 1][pos_x];
                
                for (int i = 0; i < bpp; i++)
                {
                    unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + ((left.data[i] + up.data[i]) / 2);
                }
            }
        break;
        case 4:
            
            if (pos_y < 1)
                unfiltered[x_trad] = scanline[x];
            else
            {
                raw_pixel left = (pos_x > 0) ? unfiltered[pos_y][pos_x -1] : (raw_pixel){0};
                raw_pixel up = prev_scanline[pos_y - 1][pos_x];
                raw_pixel up_left = (pos_x > 0) ? prev_scanline[pos_y - 1][pos_x - 1] : (raw_pixel){0};
                
                for (int i = 0; i < bpp; i++)
                {
                    int p = left.data[i] + up.data[i] - up_left.data[i];
                    int pa = abs(p - left.data[i]);
                    int pb = abs(p - up.data[i]);
                    int pc = abs(p - up_left.data[i]);
                    
                    if (pa <= pb && pa <= pc)
                        unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + left.data[i];
                    else if (pb <= pc)
                        unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + up.data[i];
                    else
                        unfiltered[pos_y][pos_x].data[i] = scanline[y][x].data[i] + up_left.data[i];
                }
            }
            break;
        default:
            unfiltered[pos_y][pos_x] = scanline[y][x];
            break;
    }
}

extern "C" raw_pixel *wrapper_filter(raw_pixel *whole_img, raw_pixel *diag_img,int width, int bpp,int height, int filter)
{
    raw_pixel *unfiltered;
    raw_pixel *whole_img2;
    raw_pixel *diag_img2;
    raw_pixel *ret = (raw_pixel *)malloc(height + 2 *sizeof(raw_pixel));
    cudaMalloc(&whole_img2, (height * width + 1) * sizeof(raw_pixel));
    cudaMalloc(&diag_img2, (height * width + 1) * sizeof(raw_pixel));
    cudaMemcpy(whole_img2, whole_img, (height * width) * sizeof(raw_pixel), cudaMemcpyHostToDevice);
    cudaMemcpy(diag_img2, diag_img, (height * width) * sizeof(raw_pixel), cudaMemcpyHostToDevice);
    filter_scanline<<<height, width>>>(scanline2, prev_scanline2, unfiltered, width, bpp, filter);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, unfiltered, width * bpp + 1, cudaMemcpyDeviceToHost);
    cudaFree(unfiltered);
    cudaFree(scanline2);
    cudaFree(prev_scanline2);
    return ret;
}