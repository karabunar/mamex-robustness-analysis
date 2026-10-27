"""Visual Grounding Regularization (VGR) for the two-way modality gate."""
import torch
import torch.nn.functional as F

_EPS = 1e-8


def cross_modal_cosine(image_emb, text_emb):
    """Per-item cosine similarity of the two modality embeddings; carries no gradient."""
    return F.cosine_similarity(image_emb.detach(), text_emb.detach(), dim=1, eps=_EPS)


VGR_CONTROLS = ('cosine', 'shuffled', 'constant')


def vgr_reweight(gate_weights, image_emb, text_emb, gamma, control='cosine'):
    """Down-weight the textual expert when text disagrees with the visual evidence.

    gate_weights: [N, 2] rows [w_v, w_t] summing to 1.
    w_t_reg = w_t * sigmoid(gamma * cos(v, t)), then both weights are renormalized to sum to 1.

    control isolates where the benefit comes from:
      'cosine'   - the method itself (per-item factor);
      'shuffled' - the same factors, but each item receives another item's factor (fixed permutation);
      'constant' - every item receives the mean factor.
    """
    if control not in VGR_CONTROLS:
        raise ValueError(f'control must be one of {VGR_CONTROLS}, got {control!r}')
    factor = torch.sigmoid(gamma * cross_modal_cosine(image_emb, text_emb))
    if control == 'shuffled':
        generator = torch.Generator().manual_seed(0)
        factor = factor[torch.randperm(len(factor), generator=generator).to(factor.device)]
    elif control == 'constant':
        factor = factor.mean().expand_as(factor)
    w_v, w_t = gate_weights[:, 0], gate_weights[:, 1]
    w_t_reg = w_t * factor
    total = w_v + w_t_reg
    return torch.stack([w_v / total, w_t_reg / total], dim=1)
