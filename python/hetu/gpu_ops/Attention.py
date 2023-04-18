from __future__ import absolute_import
import torch
import numpy as np
import math

from ..ndarray import array, empty, cpu
from .Node import Op
from ..gpu_links import layer_normalization, matmul_with_bias, matmul_qkv, \
                        fused_multi_head_attention, matrix_elementwise_add, \
                        gather_for_linear, scatter_for_linear, scatter_add_for_linear, matrix_slice_simple

class Attention(Op):
    def __init__(self, node_A, attention, state, encoder_hidden_states=None, config=None, ctx=None):
        self.mask = None
        self.limit_1 = None
        self.limit_2 = None
        self.config = config
        self.is_cross_attn = (encoder_hidden_states != None)
        if self.is_cross_attn:
            super().__init__(Attention, [node_A, encoder_hidden_states], ctx)
        else:
            super().__init__(Attention, [node_A], ctx)
        self.round = 0
        self.outdeg = 0
        self.latent_scale = 0
        self.built = False
        self.use_sparse = False

        self.index = None
        self.ln_mean = None
        self.ln_var = None

        if self.is_cross_attn:

            assert state == 2
            self.ln_weights = array(attention.norm2.weight, ctx=ctx)
            self.ln_bias = array(attention.norm2.bias, ctx=ctx)
            self.ln_eps = attention.norm2.eps

            self.attn_weights_q = array(attention.attn2.to_q.weight, ctx=ctx)
            self.attn_weights_k = array(attention.attn2.to_k.weight, ctx=ctx)
            self.attn_weights_v = array(attention.attn2.to_v.weight, ctx=ctx)
            self.attn_bias_q = empty((attention.attn2.to_q.out_features, ), ctx=ctx)
            self.attn_bias_k = empty((attention.attn2.to_k.out_features, ), ctx=ctx)
            self.attn_bias_v = empty((attention.attn2.to_v.out_features, ), ctx=ctx)

            self.attn_heads = attention.attn2.heads

            self.attn_to_out_weights = array(attention.attn2.to_out[0].weight, ctx=ctx)
            self.attn_to_out_bias = array(attention.attn2.to_out[0].bias, ctx=ctx)

        else:

            assert state == 1
            self.ln_weights = array(attention.norm1.weight, ctx=ctx)
            self.ln_bias = array(attention.norm1.bias, ctx=ctx)
            self.ln_eps = attention.norm1.eps

            self.attn_weights_qkv = array(torch.cat((attention.attn1.to_q.weight, 
                                                    attention.attn1.to_k.weight,
                                                    attention.attn1.to_v.weight), 0), ctx=ctx)
            self.attn_bias_qkv = empty((attention.attn1.to_q.out_features +
                                        attention.attn1.to_k.out_features +
                                        attention.attn1.to_v.out_features, ), ctx=ctx)
            assert attention.attn1.to_q.out_features == attention.attn1.to_k.out_features
            assert attention.attn1.to_k.out_features == attention.attn1.to_v.out_features
            self.gpu_buf_q = None
            self.gpu_buf_k = None
            self.gpu_buf_v = None

            self.attn_weights_q = array(attention.attn1.to_q.weight, ctx=ctx)
            self.attn_weights_k = array(attention.attn1.to_k.weight, ctx=ctx)
            self.attn_weights_v = array(attention.attn1.to_v.weight, ctx=ctx)
            self.attn_bias_q = empty((attention.attn1.to_q.out_features, ), ctx=ctx)
            self.attn_bias_k = empty((attention.attn1.to_k.out_features, ), ctx=ctx)
            self.attn_bias_v = empty((attention.attn1.to_v.out_features, ), ctx=ctx)

            self.attn_heads = attention.attn1.heads

            self.attn_to_out_weights = array(attention.attn1.to_out[0].weight, ctx=ctx)
            self.attn_to_out_bias = array(attention.attn1.to_out[0].bias, ctx=ctx)
        
        # sparse config
        self.d2h_stream = None
        # self.output_cache = []


    def build_sparse(self, input_vals, output_val):
        if self.built:
            return 
        self.built = True
        ctx = self.ctx
        if self.mask == None:
            mask = torch.load(f'runtime/mask.pt')
        else:
            mask = self.mask
        B, L, input_channel = input_vals[0].shape
        output_channel = output_val.shape[-1]
        width = int(math.sqrt(output_val.shape[-2]))
        rate = 96 // width
        mask = torch.nn.functional.interpolate(
            mask.repeat(B, 1, 1, 1), size=(96, 96)
        )
        mask = torch.nn.MaxPool2d(kernel_size=rate)(mask.float())
        mask = (mask > 0.5)
        mask = mask.numpy().reshape(-1)
        index = np.where(mask == True)
        self.index = array(index[0], ctx=ctx)
        BL_sparse = (B, self.index.shape[0] // B)
        self.input_sparse = empty(BL_sparse + (input_channel, ), ctx=ctx)
        self.q_sparse = empty(BL_sparse + (self.attn_weights_q.shape[0], ), ctx=ctx)
        if self.is_cross_attn:
            self.k_sparse = None
            self.v_sparse = None
        else:
            self.k_sparse = empty(BL_sparse + (self.attn_weights_k.shape[0], ), ctx=ctx)
            self.v_sparse = empty(BL_sparse + (self.attn_weights_v.shape[0], ), ctx=ctx)
            self.qkv_sparse = empty(BL_sparse + (self.attn_weights_qkv.shape[0], ), ctx=ctx)
        self.hidden_states_sparse = empty(BL_sparse + (self.attn_weights_v.shape[0], ), ctx=ctx)
        self.output_sparse = empty(BL_sparse + (self.attn_to_out_weights.shape[0], ), ctx=ctx)


    def compute_sparse(self, input_vals, output_val, stream_handle=None):
        self.build_sparse(input_vals, output_val)

        # Gather + LayerNorm: input -> input_sparse.
        gather_for_linear(input_vals[0], self.index, self.input_sparse, self.ln_weights, 
                         self.ln_bias, self.ln_eps, stream=stream_handle)
        # Linear: input_sparse -> q_sparse, k_sparse, v_sparse.
        # CrossAttn
        if self.is_cross_attn:
            matmul_with_bias(self.input_sparse, False, self.attn_weights_q, True, 
                            self.attn_bias_q, self.q_sparse, stream=stream_handle)
            # Reuse k and v.
            if self.k_sparse == None:
                self.k_sparse = self.k
                self.v_sparse = self.v
                matmul_with_bias(input_vals[1], False, self.attn_weights_k, True, 
                                self.attn_bias_k, self.k_sparse, stream=stream_handle)
                matmul_with_bias(input_vals[1], False, self.attn_weights_v, True, 
                                self.attn_bias_v, self.v_sparse, stream=stream_handle)
        # SelfAttn
        else:
            '''
            matmul_with_bias(self.input_sparse, False, self.attn_weights_q, True, 
                            self.attn_bias_q, self.q_sparse, stream=stream_handle)
            matmul_with_bias(self.input_sparse, False, self.attn_weights_k, True, 
                            self.attn_bias_k, self.k_sparse, stream=stream_handle)
            matmul_with_bias(self.input_sparse, False, self.attn_weights_v, True, 
                            self.attn_bias_v, self.v_sparse, stream=stream_handle)
            '''
            # Fuse qkv matmul.
            matmul_qkv(self.input_sparse, False, self.attn_weights_qkv, True, self.attn_bias_qkv,
                            self.qkv_sparse, self.q_sparse, self.k_sparse, self.v_sparse, stream=stream_handle)
            
            '''
            matmul_with_bias(self.input_sparse, False, self.attn_weights_qkv, True, 
                            self.attn_bias_qkv, self.qkv_sparse, stream=stream_handle)
            if self.gpu_buf_q == None:
                self.gpu_buf_q = array(
                    [0, 0, 0, self.qkv_sparse.shape[0], self.qkv_sparse.shape[1], self.qkv_sparse.shape[2], -1, -1, self.attn_bias_q.shape[0]],
                    ctx=self.ctx, data_type=np.uintc
                    )
                self.gpu_buf_k = array(
                    [0, 0, self.attn_bias_q.shape[0], self.qkv_sparse.shape[0], self.qkv_sparse.shape[1], self.qkv_sparse.shape[2], -1, -1, self.attn_bias_k.shape[0]],
                    ctx=self.ctx, data_type=np.uintc
                    )
                self.gpu_buf_v = array(
                    [0, 0, self.attn_bias_q.shape[0] + self.attn_bias_k.shape[0], self.qkv_sparse.shape[0], self.qkv_sparse.shape[1], self.qkv_sparse.shape[2], -1, -1, self.attn_bias_v.shape[0]],
                    ctx=self.ctx, data_type=np.uintc
                    )
            matrix_slice_simple(self.qkv_sparse, self.q_sparse, self.gpu_buf_q, stream=stream_handle)
            matrix_slice_simple(self.qkv_sparse, self.k_sparse, self.gpu_buf_k, stream=stream_handle)
            matrix_slice_simple(self.qkv_sparse, self.v_sparse, self.gpu_buf_v, stream=stream_handle)
            '''
        # Fused Attn: q_sparse, k_sparse, v_sparse -> hidden_states_sparse.
        fused_multi_head_attention(self.q_sparse, self.k_sparse, self.v_sparse,
                                  self.hidden_states_sparse, self.attn_heads, stream=stream_handle)
        # Linear: hidden_states_sparse -> output_sparse.
        matmul_with_bias(self.hidden_states_sparse, False, self.attn_to_out_weights, True, 
                        self.attn_to_out_bias, self.output_sparse, stream=stream_handle)
        # Scatter + Add: output_sparse + input -> output.
        scatter_add_for_linear(self.output_sparse, input_vals[0], output_val, self.index, stream=stream_handle)

        self.round += 1


    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        if self.latent_scale == 0:
            self.latent_scale = input_vals[0].shape[1]
        if self.use_sparse and self.round >= self.limit_1 and self.round < self.limit_2 and self.latent_scale >= self.config.latent_scale:
            self.compute_sparse(input_vals, output_val, stream_handle)
            return 

        # LayerNorm: input -> input_buffer.
        if self.ln_mean is None:
            self.ln_mean = empty(input_vals[0].shape[:2], ctx=self.ctx)
            self.ln_var = empty(input_vals[0].shape[:2], ctx=self.ctx)
        layer_normalization(input_vals[0], self.ln_weights, self.ln_bias,
                            self.ln_mean, self.ln_var, self.input_buffer, self.ln_eps, stream=stream_handle)
        # Linear: input_buffer -> q, k, v.
        # CrossAttn
        if self.is_cross_attn:
            matmul_with_bias(self.input_buffer, False, self.attn_weights_q, True, 
                            self.attn_bias_q, self.q, stream=stream_handle)
            matmul_with_bias(input_vals[1], False, self.attn_weights_k, True, 
                            self.attn_bias_k, self.k, stream=stream_handle)
            matmul_with_bias(input_vals[1], False, self.attn_weights_v, True, 
                            self.attn_bias_v, self.v, stream=stream_handle)
        # SelfAttn
        else:
            '''
            matmul_with_bias(self.input_buffer, False, self.attn_weights_q, True, 
                            self.attn_bias_q, self.q, stream=stream_handle)
            matmul_with_bias(self.input_buffer, False, self.attn_weights_k, True, 
                            self.attn_bias_k, self.k, stream=stream_handle)
            matmul_with_bias(self.input_buffer, False, self.attn_weights_v, True, 
                            self.attn_bias_v, self.v, stream=stream_handle)
            '''
            # Fuse qkv matmul.
            matmul_qkv(self.input_buffer, False, self.attn_weights_qkv, True, self.attn_bias_qkv,
                            self.qkv, self.q, self.k, self.v, stream=stream_handle)
        # Fused Attn: q, k, v -> hidden_states.
        fused_multi_head_attention(self.q, self.k, self.v,
                                  self.hidden_states, self.attn_heads, stream=stream_handle)
        # Linear: hidden_states -> output.
        matmul_with_bias(self.hidden_states, False, self.attn_to_out_weights, True, 
                        self.attn_to_out_bias, output_val, stream=stream_handle)
        # Add: output + input -> output.
        matrix_elementwise_add(output_val, input_vals[0], output_val, stream=stream_handle)
        
        # No need to save checkpoints.

        self.round += 1
        # print (self.is_cross_attn, self.round)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        B, L, C = input_shapes[0]
        self.input_buffer = empty(input_shapes[0], ctx=self.ctx)
        self.q = empty((B, L, self.attn_weights_q.shape[0]), ctx=self.ctx)
        if self.is_cross_attn:
            self.k = empty((input_shapes[1][0], input_shapes[1][1], self.attn_weights_k.shape[0]), ctx=self.ctx)
            self.v = empty((input_shapes[1][0], input_shapes[1][1], self.attn_weights_v.shape[0]), ctx=self.ctx)
        else:
            self.k = empty((B, L, self.attn_weights_k.shape[0]), ctx=self.ctx)
            self.v = empty((B, L, self.attn_weights_v.shape[0]), ctx=self.ctx)
            self.qkv = empty((B, L, self.attn_weights_qkv.shape[0]), ctx=self.ctx)
        self.hidden_states = empty((B, L, self.attn_weights_v.shape[0]), ctx=self.ctx)
        output_shape = (B, L, self.attn_to_out_weights.shape[0])
        return output_shape


def attention(node_A, attention, state=1, encoder_hidden_states=None, config=None, ctx=None):
    """Fused attention node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    attention : Torch.nn.Module
        Pytorch module.
    config :
        Hetu unet config.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Attention(node_A, attention, state=state, encoder_hidden_states=encoder_hidden_states, config=config, ctx=ctx)
