import unittest
import torch
from einops import repeat
from block_sparse_attn import block_sparse_attn_func
from utils import (
    generate_random_padding_mask,
    generate_base_sparsity_mask,
    generate_qkv,
    generate_streaming_mask,
    prepare_mixed_exact_mask,
    prepare_mixed_mask,
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S,
    get_dropout_fraction,
    attention_blocksparse_ref
)

MAX_HEADDIM_SM8x = 192
block_size = 64
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

class TestFlashAttnVarlenBlockOutput(unittest.TestCase):

    def setUp(self):
        self.device = "cuda:0"
        self.batch_size = 1
        self.seqlen_q = 128
        self.seqlen_k = 128
        self.d = 128
        self.p_dropout = 0.0
        self.causal = False
        self.exact_streaming = False
        self.sink_num = 1
        self.local_num = 3
        self.mha_type = "gqa"
        self.dtype = torch.float16
        self.sparsity = 0.0
        self.nheads = 16

    def run_test(self):
        if (
            max(self.seqlen_q, self.seqlen_k) >= 2048
            and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
        ):
            self.skipTest("Reference implementation OOM")
        
        torch.random.manual_seed(42)
        nheads_k = self.nheads if self.mha_type == "mha" else (1 if self.mha_type == "mqa" else 8)
        self.assertTrue(self.nheads % nheads_k == 0)
        window_size = (-1, -1)
        q = torch.randn(self.batch_size, self.seqlen_q, self.nheads, self.d, device=self.device, dtype=self.dtype, requires_grad=True)
        k = torch.randn(self.batch_size, self.seqlen_k, nheads_k, self.d, device=self.device, dtype=self.dtype, requires_grad=True)
        v = torch.randn(self.batch_size, self.seqlen_k, nheads_k, self.d, device=self.device, dtype=self.dtype, requires_grad=True)

        q_ref = q.view(self.batch_size, self.seqlen_q, self.nheads, self.d)
        k_ref = k.view(self.batch_size, self.seqlen_k, nheads_k, self.d)
        v_ref = v.view(self.batch_size, self.seqlen_k, nheads_k, self.d)

        query_padding_mask = generate_random_padding_mask(self.seqlen_q, self.batch_size, self.device, mode="random")
        key_padding_mask = generate_random_padding_mask(self.seqlen_k, self.batch_size, self.device, mode="random")

        alibi_slopes, attn_bias = None, None
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        # ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        ) = generate_qkv(q, k, v)


        print(k_unpad.shape)
        # q_unpad = q.view(self.batch_size, self.seqlen_q, self.nheads, self.d).transpose(1, 2)
        # k_unpad = k.view(self.batch_size, self.seqlen_k, self.nheads, self.d).transpose(1, 2)
        # v_unpad = v.view(self.batch_size, self.seqlen_k, self.nheads, self.d).transpose(1, 2)


        # q_unpad = q_unpad.transpose(1, 2).view(self.seqlen_q, self.nheads, self.d)
        # k_unpad = k_unpad.transpose(1, 2).view(self.seqlen_k, self.nheads, self.d)
        # v_unpad = v_unpad.transpose(1, 2).view(self.seqlen_k, self.nheads, self.d)

        cu_seqlens_q = torch.tensor([0, self.seqlen_q], device=self.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, self.seqlen_k], device=self.device, dtype=torch.int32)

        max_seqlen_q = self.seqlen_q
        max_seqlen_k = self.seqlen_k

        num_streaming_heads = 0
        num_blocksparse_heads = self.nheads
        num_dense_heads = 0

        sparsity_list = [self.sparsity] * num_blocksparse_heads
        head_mask_type = torch.tensor([1] * num_blocksparse_heads, device=self.device, dtype=torch.int32)
        base_blockmask = generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, block_size, block_size, block_size, self.batch_size, num_blocksparse_heads, sparsity_list, causal=self.causal, device=self.device)

        streaming_info = torch.tensor([self.sink_num, self.local_num] * self.nheads, device=self.device, dtype=torch.int32)
        streaming_mask = generate_streaming_mask(max_seqlen_q, max_seqlen_k, self.batch_size, self.nheads, cu_seqlens_q, cu_seqlens_k, block_size, block_size, block_size, streaming_info, causal=self.causal, device=self.device)
        
        if self.exact_streaming:
            self.assertTrue(self.causal)
        print(f"exact_streaming: {self.exact_streaming}")
        if self.exact_streaming:
            mixed_mask = prepare_mixed_exact_mask(base_blockmask, streaming_info, head_mask_type, self.batch_size, self.nheads, block_size, block_size, block_size, max_seqlen_q, max_seqlen_k, q.shape[1], k.shape[1], query_padding_mask, key_padding_mask, device=self.device)
        else:
            mixed_mask = prepare_mixed_mask(base_blockmask, streaming_mask, head_mask_type, self.batch_size, self.nheads, block_size, block_size, block_size, max_seqlen_q, max_seqlen_k, q.shape[1], k.shape[1], device=self.device)
        
        print(q_unpad.shape, k_unpad.shape, v_unpad.shape)

        out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type,
            None,
            base_blockmask,
            max_seqlen_q, max_seqlen_k,
            self.p_dropout,
            deterministic=True,
            softmax_scale=None,
            is_causal=self.causal,
            exact_streaming=self.exact_streaming,
            return_attn_probs=True,
        )
        print(out_unpad.shape)
        
        # out = output_pad_fn(out_unpad)
        out = out_unpad.view(self.batch_size, self.seqlen_q, self.nheads, self.d).transpose(1, 2)

        out = out.transpose(1, 2).contiguous()
        print(out.shape)
        
        if self.p_dropout > 0.0:
            self.assertIsNotNone(S_dmask)
            S_dmask_converted = convert_flash_attn_S_to_softmax(
                S_dmask,
                self.seqlen_q,
                self.seqlen_k,
                None,
                None,
                self.d,
                self.p_dropout > 0.0,
                causal=self.causal,
                window_size=window_size,
            )
            dropout_mask = S_dmask_converted >= 0
            attn_unnorm = S_dmask_converted.abs()
            
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=self.nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=self.nheads // nheads_k)
            
            attn = normalize_flash_attn_S(
                attn_unnorm,
                q_ref,
                k_rep,
                v_rep,
                None,
                None,
                attn_bias,
                self.p_dropout > 0.0,
                causal=self.causal,
                window_size=window_size,
            )
            
            dropout_fraction = get_dropout_fraction(
                dropout_mask,
                mixed_mask,
                block_size, block_size, 
                None,
                None,
                causal=self.causal,
                window_size=window_size,
            ).item()
            
            print(f"Actual dropout fraction: {dropout_fraction}")
        else:
            dropout_mask = None

        out_ref, attn_ref = attention_blocksparse_ref(
                q_ref,
                k_ref,
                v_ref,
                mixed_mask,
                block_size, block_size, 
                None,
                None,
                self.p_dropout,
                dropout_mask,
                causal=self.causal,
                window_size=window_size,
            )
        out_pt, attn_pt = attention_blocksparse_ref(
                q_ref,
                k_ref,
                v_ref,
                mixed_mask,
                block_size, block_size, 
                None,
                None,
                self.p_dropout,
                dropout_mask,
                causal=self.causal,
                window_size=window_size,
                upcast=False,
                reorder_ops=True,
            )

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

        self.assertLessEqual((out - out_ref).abs().max().item(), 2 * (out_pt - out_ref).abs().max().item())

    def test_case_1(self):
        self.sparsity = 0.5
        self.run_test()

    # def test_case_2(self):
    #     self.seqlen_q = 113
    #     self.seqlen_k = 203
    #     self.sparsity = 0.5

    #     self.run_test()

    # def test_case_3(self):
    #     self.seqlen_q = 128
    #     self.seqlen_k = 217
    #     self.sparsity = 0.5
    #     self.run_test()

    # def test_case_4(self):
    #     self.seqlen_q = 113
    #     self.seqlen_k = 211
    #     self.run_test()

    # def test_case_5(self):
    #     self.seqlen_q = 108
    #     self.seqlen_k = 256
    #     self.run_test()

    # def test_case_6(self):
    #     self.seqlen_q = 256
    #     self.seqlen_k = 512
    #     self.run_test()

    # def test_case_7(self):
    #     self.seqlen_q = 512
    #     self.seqlen_k = 256
    #     self.run_test()

    # def test_case_8(self):
    #     self.seqlen_q = 1024
    #     self.seqlen_k = 1024
    #     self.run_test()

    # def test_case_9(self):
    #     self.seqlen_q = 1023
    #     self.seqlen_k = 1024
    #     self.run_test()

    # def test_case_10(self):
    #     self.seqlen_q = 1024
    #     self.seqlen_k = 1023
    #     self.run_test()

    # def test_case_11(self):
    #     self.seqlen_q = 2048
    #     self.seqlen_k = 2048
    #     self.run_test()

if __name__ == '__main__':
    unittest.main()
