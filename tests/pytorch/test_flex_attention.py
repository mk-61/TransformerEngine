from functools import partial
import math

import cudnn
import torch
from torch.nn.attention.flex_attention import flex_attention

import pytest


SOFTCAP = 20
NEG_INF = float("-inf")


def causal_mask_torch(score, _b, _h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def causal_mask_graph(sdpa_graph, q_kt_tensor, varpack):
    row_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=2).set_data_type(cudnn.data_type.INT32)
    col_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=3).set_data_type(cudnn.data_type.INT32)

    mask = sdpa_graph.cmp_ge(
        input=row_index, comparison=col_index, compute_data_type=cudnn.data_type.BOOLEAN
    ).set_data_type(cudnn.data_type.BOOLEAN)

    neg_inf_tensor_cpu = torch.full((1, 1, 1, 1), NEG_INF)
    neg_inf_tensor = sdpa_graph.tensor(
        name="neg_inf_tensor",
        dim=neg_inf_tensor_cpu.size(),
        stride=neg_inf_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_cpu.dtype,
    )
    varpack[neg_inf_tensor] = neg_inf_tensor_cpu
    return sdpa_graph.binary_select(input0=neg_inf_tensor, input1=q_kt_tensor, mask=mask)


def relative_positional_torch(score, _b, _h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


def relative_positional_graph(sdpa_graph, q_kt_tensor, varpack):
    row_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=2)
    col_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=3)
    sub_out = sdpa_graph.sub(a=row_index, b=col_index)
    return sdpa_graph.add(a=q_kt_tensor, b=sub_out) 


def softcap_torch(score, _b, _h, _q_idx, _kv_idx):
    return torch.tanh(score / SOFTCAP) * SOFTCAP


def softcap_graph(sdpa_graph, q_kt_tensor, varpack):
    softcap_tensor_cpu = torch.full((1, 1, 1, 1), SOFTCAP)
    softcap_tensor = sdpa_graph.tensor(
        name="softcap_tensor",
        dim=softcap_tensor_cpu.size(),
        stride=softcap_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=softcap_tensor_cpu.dtype,
    )
    varpack[softcap_tensor] = softcap_tensor_cpu
    div_out = sdpa_graph.div(a=q_kt_tensor, b=softcap_tensor)
    tanh_out = sdpa_graph.tanh(input=div_out)
    return sdpa_graph.mul(a=tanh_out, b=softcap_tensor)


@pytest.mark.parametrize("score_mod,score_mod_ref", [
    (None, None),
    (causal_mask_graph, causal_mask_torch),
    (relative_positional_graph, relative_positional_torch),
    (softcap_graph, softcap_torch),
])
@pytest.mark.parametrize("b,h,s,d", [(4, 12, 1024, 64)])
def test_score_mod(score_mod, score_mod_ref, b, h, s, d):
    attn_scale = 1.0 / math.sqrt(d)
    strides = s * h * d, d, h * d, 1

    q_gpu = torch.randn(b * s * h * d).half().cuda().as_strided((b, h, s, d), strides)
    k_gpu = torch.randn(b * s * h * d).half().cuda().as_strided((b, h, s, d), strides)
    v_gpu = torch.randn(b * s * h * d).half().cuda().as_strided((b, h, s, d), strides)
    o_gpu = torch.empty(b * s * h * d).half().cuda().as_strided((b, h, s, d), strides)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    varpack_extra = {}
    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=True,
        attn_scale=attn_scale,
        use_causal_mask=False,
        score_mod=partial(score_mod, varpack=varpack_extra) if score_mod else None,
    )

    o.set_output(True).set_dim((b, h, s, d)).set_stride(strides)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
    }
    variant_pack.update(varpack_extra)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    q_ref = q_gpu.detach().float().requires_grad_()
    k_ref = k_gpu.detach().float().requires_grad_()
    v_ref = v_gpu.detach().float().requires_grad_()

    o_ref = flex_attention(q_ref, k_ref, v_ref, score_mod=score_mod_ref)
    torch.testing.assert_close(o_ref, o_gpu.float(), atol=5e-3, rtol=3e-3)

