#include <stdlib.h>
#include <stdio.h>


__global__ void filter_scanline(unsigned char *scanline, unsigned char *prev_scanline, unsigned char *unfiltered, int width, int bpp, int filter)
{
    int x  = threadIdx.x + blockDim.x * blockIdx.x;
    
    int x_trad = x;
    switch (filter)
    {
        case 0:
                unfiltered[x_trad] = scanline[x];
            break;
        case 1:
            if (x <= bpp)
                unfiltered[x_trad] = scanline[x];
            else
                unfiltered[x_trad] = scanline[x] + unfiltered[x_trad - bpp];
            break;
        case 2:
            if (prev_scanline)
                unfiltered[x_trad] = scanline[x];
            else
                unfiltered[x_trad] = scanline[x] + prev_scanline[x_trad - 1];
            break;
        case 3:
            if (prev_scanline)
                unfiltered[x_trad] = scanline[x];
            else
            {
                int left = (x > bpp) ? unfiltered[x_trad - bpp] : 0;
                int up = prev_scanline[x_trad];
                unfiltered[x_trad] = scanline[x] + ((left + up) / 2);
            }
        break;
        case 4:
            
            if (prev_scanline)
                unfiltered[x_trad] = scanline[x];
            else
            {
                int left = (x > bpp) ? unfiltered[x_trad - bpp] : 0;
                int up = prev_scanline[x_trad];
                int up_left = (x > bpp) ? prev_scanline[x_trad - bpp] : 0;
                
                int p = left + up - up_left;
                int pa = abs(p - left);
                int pb = abs(p - up);
                int pc = abs(p - up_left);
                
                if (pa <= pb && pa <= pc)
                    unfiltered[x_trad] = scanline[x] + left;
                else if (pb <= pc)
                    unfiltered[x_trad] = scanline[x] + up;
                else
                    unfiltered[x_trad] = scanline[x] + up_left;
            }
            break;
        default:
            unfiltered[x-1] = scanline[x];
            break;
    }
}

extern "C" unsigned char *wrapper_filter(unsigned char *scanline, unsigned char *prev_scanline, int width, int bpp, int filter)
{
    unsigned char *unfiltered;
    unsigned char *scanline2;
    unsigned char *prev_scanline2;
    unsigned char *ret = (unsigned char *)malloc(width * bpp + 2);
    cudaMalloc(&unfiltered, width * bpp + 2);
    cudaMalloc(&scanline2, width * bpp + 2);
    cudaMalloc(&prev_scanline2, width * bpp + 2);

    cudaMemcpy(scanline2, scanline, width * bpp + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(prev_scanline2, prev_scanline, width * bpp + 1, cudaMemcpyHostToDevice);
    scanline2++;
    prev_scanline2++;
    filter_scanline<<<1, width * bpp>>>(scanline2, prev_scanline2, unfiltered, width, bpp, filter);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, unfiltered, width * bpp, cudaMemcpyDeviceToHost);
    cudaFree(unfiltered);
    cudaFree(scanline2);
    cudaFree(prev_scanline2);
    return ret;
}