#include <gpu_runtime.h>
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma.h"
#include "xformers/kernel_forward.h"

struct Config {
    int dev_id;
    int64_t num_batches, num_heads;
    int64_t query_seq_len, kv_seq_len;
    int64_t head_size, value_head_size;
    int64_t q_stride_h, q_stride_m, q_stride_b;
    int64_t k_stride_h, k_stride_m, k_stride_b;
    int64_t v_stride_h, v_stride_m, v_stride_b;
    const void* query_ptr;
    const void* key_ptr;
    const void* value_ptr;
    void* out_ptr;
};

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration>
void BuildCutlassKernel(const Config& config, DLStreamHandle stream) {
    using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                                      single_value_iteration, false, false>;
    typename Attention::Params p{};
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(config.query_ptr));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(config.key_ptr));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(config.value_ptr));
    p.attn_bias_ptr = nullptr;
    p.logsumexp_ptr = nullptr;
    p.output_ptr = reinterpret_cast<T*>(config.out_ptr);
    void *workspace = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
        using Acc = typename Attention::accum_t;
        workspace = find_chunk(config.num_batches * config.query_seq_len * config.num_heads
                              * config.value_head_size * sizeof(Acc), config.dev_id);
        p.output_accum_ptr = reinterpret_cast<Acc*>(workspace);
    } else {
        p.output_accum_ptr = nullptr;
    }
    p.num_heads = config.num_heads;
    p.num_batches = config.num_batches;
    p.head_dim = config.head_size;
    p.head_dim_value = config.value_head_size;
    p.num_queries = config.query_seq_len;
    p.num_keys = config.kv_seq_len;
    p.q_strideM = config.q_stride_m;
    p.k_strideM = config.k_stride_m;
    p.v_strideM = config.v_stride_m;
    p.o_strideM = p.head_dim_value * p.num_heads;

    p.q_strideH = config.q_stride_h;
    p.k_strideH = config.k_stride_h;
    p.v_strideH = config.v_stride_h;

    p.q_strideB = config.q_stride_b;
    p.k_strideB = config.k_stride_b;
    p.v_strideB = config.v_stride_b;

    p.scale = 1.0f / std::sqrt(float(p.head_dim));

    p.causal = false;
    p.causal_diagonal_offset = 0;
    p.use_dropout = false;

    assert(Attention::check_supported(p) == true);

    constexpr auto kernel = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (stream)
        kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), 
                  smem_bytes, *(cudaStream_t *)stream->handle>>>(p);
    else
        kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);

    if (workspace)
        del_chunk(workspace, config.dev_id);
}

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block>
void WrapSingleValueIteration(const Config& config, DLStreamHandle stream) {
    if (config.value_head_size <= keys_per_block) {
        BuildCutlassKernel<T, ArchTag, is_aligned, 
                            queries_per_block, keys_per_block, true>(config, stream);
    } else {
        BuildCutlassKernel<T, ArchTag, is_aligned, 
                            queries_per_block, keys_per_block, false>(config, stream);
    }
}

template<typename T, typename ArchTag, bool is_aligned>
void WrapKeysPerBlock(const Config& config, DLStreamHandle stream) {
    if (config.value_head_size <= 64) {
        WrapSingleValueIteration<T, ArchTag, is_aligned, 64, 64>(config, stream);
    } else {
        WrapSingleValueIteration<T, ArchTag, is_aligned, 32, 128>(config, stream);
    }
}

template<typename T, typename ArchTag>
void WrapIsAligned(const Config& config, DLStreamHandle stream) {
    if (reinterpret_cast<uintptr_t>(config.query_ptr) % 16 == 0
        && reinterpret_cast<uintptr_t>(config.key_ptr) % 16 == 0
        && reinterpret_cast<uintptr_t>(config.value_ptr) % 16 == 0
        && config.head_size % (16 / sizeof(T)) == 0
        && config.value_head_size % (16 / sizeof(T)) == 0) {
        WrapKeysPerBlock<T, ArchTag, true>(config, stream);
    } else {
        WrapKeysPerBlock<T, ArchTag, false>(config, stream);
    }
}

template<typename T>
void WrapArchTag(const Config& config, DLStreamHandle stream, int dev_id) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    if (deviceProp.major == 7) {
        if (deviceProp.minor == 5)
            WrapIsAligned<T, cutlass::arch::Sm75>(config, stream);
        else
            WrapIsAligned<T, cutlass::arch::Sm70>(config, stream);
    } else if (deviceProp.major == 8) {
        WrapIsAligned<T, cutlass::arch::Sm80>(config, stream);
    }
    else {
        assert(false);
    }
}

int DLGpuFusedMultiHeadAttention(const DLArrayHandle query,
               const DLArrayHandle key, const DLArrayHandle value,
               DLArrayHandle output, int num_heads, DLStreamHandle stream_handle = NULL) {
    // default shape: (batch, seq_len, num_heads * head_sizes)
    assert(query->ndim == 3);
    assert(key->ndim == 3);
    assert(value->ndim == 3);
    Config config{};
    config.dev_id = (query->ctx).device_id;
    config.num_batches = query->shape[0];
    config.num_heads = num_heads;
    config.query_seq_len = query->shape[1];
    config.kv_seq_len = value->shape[1];
    config.head_size = query->shape[2] / num_heads;
    config.value_head_size = value->shape[2] / num_heads;
    config.q_stride_h = query->shape[2] / num_heads;
    config.q_stride_m = query->shape[2];
    config.q_stride_b = config.q_stride_m * query->shape[1];
    config.k_stride_h = key->shape[2] / num_heads;
    config.k_stride_m = key->shape[2];
    config.k_stride_b = config.k_stride_m * key->shape[1];
    config.v_stride_h = value->shape[2] / num_heads;
    config.v_stride_m = value->shape[2];
    config.v_stride_b = config.v_stride_m * value->shape[1];
    config.query_ptr = (const float *)query->data;
    config.key_ptr = (const float *)key->data;
    config.value_ptr = (const float *)value->data;
    config.out_ptr = (float *)output->data;
    int dev_id = (query->ctx).device_id;
    WrapArchTag<float>(config, stream_handle, dev_id);
    return 0;
}