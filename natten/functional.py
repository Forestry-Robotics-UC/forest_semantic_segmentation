import torch
import torch.nn.functional as F


def _validate_2d(kernel_size, dilation):
    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size

    if isinstance(dilation, int):
        d_h = d_w = dilation
    else:
        d_h, d_w = dilation

    if k_h != k_w:
        raise ValueError("Only square kernels are supported in this compatibility implementation.")
    if d_h != d_w:
        raise ValueError("Only equal 2D dilation is supported in this compatibility implementation.")
    return k_h, d_h


def _extract_neighborhood_2d(x, kernel_size, dilation):
    # x: [B, H, W, C]
    b, h, w, c = x.shape
    pad = dilation * (kernel_size - 1) // 2
    x_bchw = x.permute(0, 3, 1, 2).contiguous()
    unfolded = F.unfold(
        x_bchw,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=pad,
        stride=1,
    )
    k2 = kernel_size * kernel_size
    # [B, C*K2, H*W] -> [B, H, W, K2, C]
    neighborhoods = unfolded.view(b, c, k2, h, w).permute(0, 3, 4, 2, 1).contiguous()
    return neighborhoods


def _flatten_rpb(rpb, kernel_size):
    # rpb: [num_heads, 2k-1, 2k-1]
    if rpb is None:
        return None
    if rpb.dim() != 3:
        raise ValueError(f"Expected rpb with 3 dims [heads, 2k-1, 2k-1], got shape={tuple(rpb.shape)}")
    center = kernel_size - 1
    offsets = range(-(kernel_size // 2), kernel_size // 2 + 1)
    bias = []
    for dy in offsets:
        for dx in offsets:
            bias.append(rpb[:, center + dy, center + dx])
    # [K2, heads] -> [heads, K2]
    return torch.stack(bias, dim=0).transpose(0, 1).contiguous()


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """
    Compatibility implementation for transformers.models.dinat.
    query/key: [B, heads, H, W, head_dim]
    returns attention logits: [B, heads, H, W, K*K]
    """
    k, d = _validate_2d(kernel_size, dilation)
    b, heads, h, w, dim = query.shape
    q = query.view(b * heads, h, w, dim)
    k_t = key.view(b * heads, h, w, dim)
    neighborhoods = _extract_neighborhood_2d(k_t, kernel_size=k, dilation=d)
    # [BH, H, W, K2]
    scores = (q.unsqueeze(-2) * neighborhoods).sum(dim=-1)
    scores = scores.view(b, heads, h, w, k * k)

    bias = _flatten_rpb(rpb, k)
    if bias is not None:
        scores = scores + bias[:, None, None, :]
    return scores


def natten2dav(attention_probs, value, kernel_size, dilation):
    """
    Compatibility implementation for transformers.models.dinat.
    attention_probs: [B, heads, H, W, K*K]
    value: [B, heads, H, W, head_dim]
    returns context: [B, heads, H, W, head_dim]
    """
    k, d = _validate_2d(kernel_size, dilation)
    b, heads, h, w, dim = value.shape
    v = value.view(b * heads, h, w, dim)
    neighborhoods = _extract_neighborhood_2d(v, kernel_size=k, dilation=d)
    attn = attention_probs.view(b * heads, h, w, k * k)
    context = (attn.unsqueeze(-1) * neighborhoods).sum(dim=-2)
    return context.view(b, heads, h, w, dim)
