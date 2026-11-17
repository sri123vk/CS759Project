                    std::fprintf(stderr, "skip auto-block %dx%d: %s\n", bx, by, error.what());
                }
            }
        } catch (...) {
            cuda_conv_free(c);
            throw;
        }
        cuda_conv_free(c);
    });
    if (best_x == 0) {
        std::fprintf(stderr, "no supported auto-block candidate for %dx%d\n", a.H, a.W);
        std::exit(1);
    }
    a.bx = best_x;
    a.by = best_y;
    std::fprintf(stderr,
                 "auto-block %s %dx%d K=%d: %dx%d (trial kernel median %.6f ms; tuning %.3f ms, excluded from reported timings)\n",
                 a.impl.c_str(), a.H, a.W, a.K, a.bx, a.by, best_kernel, tuning_ms);
}
#endif
// "%.4f", or an empty field for NaN.
std::string field(double v) {
    if (std::isnan(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4f", v);
    return buf;
}

}  // namespace
int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const bool cuda = (a.impl == "cuda");
    const bool cuda_opt = (a.impl == "cuda_opt");
    const bool cuda_extra = is_extra_cuda(a.impl);
    const size_t n = (size_t)a.H * a.W;
    // out starts as NaN so any pixel an implementation never writes fails --check.
    std::vector<float> in(n), k((size_t)a.K * a.K),
        out(n, std::numeric_limits<float>::quiet_NaN());
    fill_random(in, 759);
    fill_random(k, 42);

    Timings t;
    try {
        if (a.auto_block) {
            if (!cuda && !cuda_opt && !cuda_extra) {
                std::fprintf(stderr, "--block auto requires a CUDA implementation\n");
                return 1;
            }
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_OPT)
            autotune_block(a, in, out, k);
#else
            std::fprintf(stderr, "--block auto requires both CUDA implementations to be built\n");
            return 1;
#endif
        }

        if (cuda) {
#ifdef CONV_HAVE_CUDA
            t = bench_cuda(a, in, out, k);
#else
            std::fprintf(stderr, "built without CUDA (nvcc not found when running make)\n");
            return 1;
#endif
        } else if (cuda_opt || cuda_extra) {
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_OPT)
            t = bench_cuda_opt(a, in, out, k);
#else
            std::fprintf(stderr, "CUDA implementation '%s' is unavailable in this build\n", a.impl.c_str());
            return 1;
#endif
        } else {
            t = bench_cpu(a, in, out, k);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA benchmark failed: %s\n", error.what());
        return 1;
    }

    ErrStats err;
    bool pass = true;
    if (a.check) {
        std::vector<float> ref(n);
        conv2d_seq(in.data(), ref.data(), k.data(), a.H, a.W, a.K);
        err = compare(ref, out, sum_abs_terms(in, k, a.H, a.W, a.K), a.K);
        pass = within_tol(err);
    }

    if (!a.dump_dir.empty()) {
        std::string base = a.dump_dir + "/" + std::to_string(a.H) + "x" +
                           std::to_string(a.W) + "_k" + std::to_string(a.K) + "_";
        if (!write_bin(base + "in.bin", in) || !write_bin(base + "k.bin", k) ||
            !write_bin(base + a.impl + "_out.bin", out)) {
            std::fprintf(stderr, "failed to write dump files to %s\n", a.dump_dir.c_str());
            return 1;
        }
    }

    const std::string bx = (cuda || cuda_opt || cuda_extra) ? std::to_string(a.bx) : "";
    const std::string by = (cuda || cuda_opt || cuda_extra) ? std::to_string(a.by) : "";
    if (a.header)
        std::printf("impl,H,W,K,bx,by,iters,median_ms,min_ms,"
                    "h2d_ms,kernel_ms,d2h_ms,alloc_ms,"
                    "max_abs_err,max_rel_err,err_ratio,nonfinite,pass\n");
    std::printf("%s,%d,%d,%d,%s,%s,%d,%.4f,%.4f,%s,%s,%s,%s,%.3e,%.3e,%.3e,%zu,%d\n",
                a.impl.c_str(), a.H, a.W, a.K, bx.c_str(), by.c_str(), a.iters,
                median(t.call_ms),
                *std::min_element(t.call_ms.begin(), t.call_ms.end()),
                field(t.h2d_ms).c_str(), field(t.kernel_ms).c_str(),
                field(t.d2h_ms).c_str(), field(t.alloc_ms).c_str(),
                err.max_abs, err.max_rel, err.max_ratio, err.nonfinite,
                pass ? 1 : 0);
    return pass ? 0 : 2;
}
