// PP-DocLayoutV3 tagged custom ops (patched for ppdoclayout, see ppdl-ops.cu)
bool ggml_cuda_compute_forward_ppdl(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
bool ggml_cuda_supports_ppdl_op(const struct ggml_tensor * op);
bool ggml_cuda_ppdl_im2col_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream);
