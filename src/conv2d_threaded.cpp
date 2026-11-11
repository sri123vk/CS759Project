// Sequential parallelized: 2D convolution using openMP, one output pixel at a time.
// This is a direct parallelization of the sequential implementation, with the outer loop parallelized.


#include "conv2d.h"
#include <thread>
#include <algorithm>


void conv2d_threaded(const float* in, float* out, const float* k,
                int H, int W, int K) {
    const int R = K / 2;


    
}
