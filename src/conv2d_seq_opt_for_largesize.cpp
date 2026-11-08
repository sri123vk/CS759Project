#include <iostream>
#include <cstddef>

#include "conv2d.h"

void convolve(
    const float *image,
    float *output,
    std::size_t height,
    std::size_t width,
    const float *mask,
    std::size_t m
) {
    using signed_size_t = std::ptrdiff_t;

    if (m == 0 || m % 2 == 0 ||
        height == 0 || width == 0 ||
        image == nullptr || output == nullptr || mask == nullptr) {
        std::cerr << "Invalid input parameters for convolution function\n";
        return;
    }

    // Signed indices let coordinates become negative at the boundary.
    const signed_size_t size_h = static_cast<signed_size_t>(height);
    const signed_size_t size_w = static_cast<signed_size_t>(width);
    const signed_size_t size_m = static_cast<signed_size_t>(m);

    const signed_size_t half_size = size_m / 2;

    // i: image row
    for (signed_size_t i = 0; i < size_h; ++i) {
        const bool row_inside = (i >= half_size && i + half_size < size_h);

        // j: image column
        for (signed_size_t j = 0; j < size_w; ++j) {
            float sum = 0.0f;

            if (row_inside &&
                j >= half_size &&
                j + half_size < size_w) {

                // Interior pixel: the whole mask fits inside.
                const float *image_start = image + (i - half_size) * size_w + (j - half_size);

                // k: mask row
                for (signed_size_t k = 0; k < size_m; ++k) {
                    const float *image_row = image_start + k * size_w;
                    const float *mask_row = mask + k * size_m;
                    // l: mask column
                    for (signed_size_t l = 0; l < size_m; ++l) {
                        sum += image_row[l] * mask_row[l];
                    }
                }
            }
            else {
                // Boundary pixel: positions outside the image
                // have value 0, so their contributions are skipped.
                for (signed_size_t k = 0; k < size_m; ++k) {
                    const signed_size_t image_i =i + k - half_size;
                    const bool row_in_bounds =(image_i >= 0 && image_i < size_h);
                    const float *mask_row = mask + k * size_m;
                    for (signed_size_t l = 0; l < size_m; ++l) {
                        const signed_size_t image_j = j + l - half_size;
                        const bool col_in_bounds = (image_j >= 0 && image_j < size_w);

                        if (row_in_bounds && col_in_bounds) {
                            sum += image[image_i * size_w + image_j]* mask_row[l];
                        }

                        // Otherwise, add nothing: image value = 0.
                    }
                }
            }

            output[i * size_w + j] = sum;
        }
    }
}