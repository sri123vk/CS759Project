// Threaded implementation used cpp <thread>: 2D convultion using threads.


#include "conv2d.h"
#include <thread>
#include <vector>
#include <algorithm>
#include "conv2d.h"


/**
 * Implementation of a 2d convolution manually
 * using threads.
 * Changes the number of threads and method of parallelization
 * depending on the size of the image and kernel.
 * For small images, there's likely little performance gain from 
 * parallelization, so few threads are needed. For larger images,
 * we can more aggressively parallelize the process and 
 * iterate over both rows and columns. Probably. Well I will. But it will
 * probably help.
 * Inputs:
 *  in: input image of size H x W
 *  out: output image of size H x W
 *  k: kernel of size K x K
 *  H: height of the input image
 *  W: width of the input image
 *  K: size of the kernel (assumed to be odd)
 */
void conv2d_threaded(const float* in, float* out, const float* k,
                int H, int W, int K) {
    
    const int R = K / 2;

    //Decide on the number of threads to use based on the size of the image and kernel
    //S.W.A.G: Each thread should have at least 1000 pixels to process; cap threads at 8.
    int num_threads = std::min(8, std::max(1, (H * W) / 1000));

    if (num_threads == 1) {
        conv2d_seq(in, out, k, H, W, K);
        return;
    }

    std::vector<std::thread> threads;

    //Divide the image into chunks for the threads
    for (int t = 0; t < num_threads; ++t) {
        int start_row = t * H / num_threads;
        int end_row = (t + 1) * H / num_threads;

        //Copilot autocomplete used to create lambda function.
        //Implementation based off of conv2d_seq
        //Create a thread for each chunk
        threads.emplace_back([=]() {
            for (int r = start_row; r < end_row; ++r) {
                // Filter rows that land inside the image for this output row.
                // Clipping the loop bounds is equivalent to zero padding but keeps
                // the bounds check out of the inner loop.
                const int i_lo = std::max(0, R - r);
                const int i_hi = std::min(K, H - r + R);

                for (int c = 0; c < W; ++c) {
                    const int j_lo = std::max(0, R - c);
                    const int j_hi = std::min(K, W - c + R);

                    float acc = 0.0f;
                    for (int i = i_lo; i < i_hi; ++i) {
                        const int y = r + i - R;
                        for (int j = j_lo; j < j_hi; ++j) {
                            acc += k[i * K + j] * in[y * W + (c + j - R)];
                        }
                    }
                    out[r * W + c] = acc;
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
}
