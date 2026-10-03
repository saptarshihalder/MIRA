"""Variable-width, support-only prediction corrections; no positional embeddings."""
import torch
from torch import nn


def _clean(x, mask):
    if x.ndim != 3 or x.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Expected x [B,N,D] and an equally shaped boolean missing mask")
    if x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError("Rows and fields must be nonempty")
    value = torch.where(mask, torch.zeros_like(x), x)
    if not torch.isfinite(value).all():
        raise ValueError("Observed values must be finite")
    return torch.tanh(value / 4.0)


def _check_rows(value, x, name):
    if value.shape != x.shape[:2] or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape [B,N]")


class _ContextEncoder(nn.Module):
    """Field interactions followed by query-specific attention to labeled support."""
    def __init__(self, width=32, heads=4):
        super().__init__()
        if width < 4 or width % heads:
            raise ValueError("width must be >=4 and divisible by heads")
        self.field = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.LayerNorm(width))
        self.field_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.field_norm = nn.LayerNorm(width)
        self.field_ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.field_ff_norm = nn.LayerNorm(width)
        self.row = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.query = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.LayerNorm(width))
        self.support = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.LayerNorm(width))
        self.support_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)

    @staticmethod
    def descriptors(x, mask, y, logit):
        value = _clean(x, mask)
        observed = (~mask).to(x.dtype)
        missing = mask.to(x.dtype)
        count = observed.sum(1).clamp_min(1)
        missing_count = missing.sum(1).clamp_min(1)
        probability = logit.sigmoid().unsqueeze(-1)
        residual = y.unsqueeze(-1) - probability
        mean = (value * observed).sum(1) / count
        variance = ((value - mean[:, None, :]).square() * observed).sum(1) / count
        return torch.stack((
            mean, (variance + 1e-6).sqrt(), missing.mean(1),
            residual.mean(1).expand_as(mean),
            (value * observed * residual).sum(1) / count,
            (missing * residual).mean(1),
            (observed * probability).sum(1) / count,
            (missing * probability).sum(1) / missing_count,
        ), dim=-1)

    def rows(self, x, mask, descriptors):
        value = _clean(x, mask)
        context = descriptors[:, None, :, :].expand(-1, x.shape[1], -1, -1)
        fields = self.field(torch.cat((value[..., None], mask.to(x.dtype)[..., None], context), -1))
        batch, rows, columns, width = fields.shape
        fields = fields.reshape(batch * rows, columns, width)
        mixed, _ = self.field_attention(fields, fields, fields, need_weights=False)
        fields = self.field_norm(fields + mixed)
        fields = self.field_ff_norm(fields + self.field_ff(fields))
        pooled = torch.cat((fields.mean(1), fields.amax(1)), -1)
        return self.row(pooled).reshape(batch, rows, width)

    def forward(self, query_x, query_mask, query_logit, support_x, support_mask, support_y, support_logit):
        _check_rows(query_logit, query_x, "query_logit")
        _check_rows(support_logit, support_x, "support_logit")
        _check_rows(support_y, support_x, "support_y")
        if query_x.shape[0] != support_x.shape[0] or query_x.shape[2] != support_x.shape[2]:
            raise ValueError("Query and support must share batch and field dimensions")
        if not ((support_y >= 0) & (support_y <= 1)).all():
            raise ValueError("Support labels must be in [0,1]")
        descriptors = self.descriptors(support_x, support_mask, support_y, support_logit)
        query = self.query(torch.cat((self.rows(query_x, query_mask, descriptors), query_logit.sigmoid()[..., None]), -1))
        residual = support_y - support_logit.sigmoid()
        support = self.support(torch.cat((self.rows(support_x, support_mask, descriptors),
                                          support_logit.sigmoid()[..., None], residual[..., None]), -1))
        attended, _ = self.support_attention(query, support, support, need_weights=False)
        pooled = support.mean(1)[:, None, :].expand(-1, query.shape[1], -1)
        return torch.cat((query, attended, pooled), -1)


class Bridge(nn.Module):
    """logit + h(query,target support) - h(query,source support)."""
    def __init__(self, width=32, heads=4, max_potential=2.0, ablation="none"):
        super().__init__()
        if max_potential <= 0:
            raise ValueError("max_potential must be positive")
        self.encoder = _ContextEncoder(width, heads)
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.max_potential = float(max_potential)
        if ablation not in ("none", "target_only", "no_mask", "no_query"):
            raise ValueError("Unknown ablation")
        self.ablation = ablation

    def potential(self, query_x, query_mask, query_logit, support_x, support_mask, support_y, support_logit):
        if self.ablation == "no_mask":
            query_x = torch.where(query_mask, torch.zeros_like(query_x), query_x)
            support_x = torch.where(support_mask, torch.zeros_like(support_x), support_x)
            query_mask, support_mask = torch.zeros_like(query_mask), torch.zeros_like(support_mask)
        elif self.ablation == "no_query":
            query_x, query_mask, query_logit = torch.zeros_like(query_x), torch.zeros_like(query_mask), torch.zeros_like(query_logit)
        representation = self.encoder(query_x, query_mask, query_logit, support_x, support_mask, support_y, support_logit)
        raw = self.head(representation).squeeze(-1)
        return self.max_potential * torch.tanh(raw / self.max_potential)

    def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                source_y, source_logit, target_x, target_mask, target_y, target_logit):
        if self.ablation == "target_only":
            return query_logit + self.potential(query_x, query_mask, query_logit, target_x, target_mask, target_y, target_logit)
        source = self.potential(query_x, query_mask, query_logit, source_x, source_mask, source_y, source_logit)
        target = self.potential(query_x, query_mask, query_logit, target_x, target_mask, target_y, target_logit)
        return query_logit + (target - source)


class GenericContextual(nn.Module):
    """Unconstrained source/target pair head with the same contextual encoder."""
    def __init__(self, width=32, heads=4, max_potential=2.0):
        super().__init__()
        self.encoder = _ContextEncoder(width, heads)
        # Halving hidden width nearly matches Bridge's head parameter budget.
        self.head = nn.Sequential(nn.Linear(6 * width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.max_potential = float(max_potential)
        if max_potential <= 0:
            raise ValueError("max_potential must be positive")

    def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                source_y, source_logit, target_x, target_mask, target_y, target_logit):
        source = self.encoder(query_x, query_mask, query_logit, source_x, source_mask, source_y, source_logit)
        target = self.encoder(query_x, query_mask, query_logit, target_x, target_mask, target_y, target_logit)
        raw = self.head(torch.cat((source, target), -1)).squeeze(-1)
        return query_logit + 2 * self.max_potential * torch.tanh(raw / (2 * self.max_potential))


class LinearResidual(nn.Module):
    """Linear query/context descriptor control; width-independent, 13 parameters."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(12, 1)

    def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                source_y, source_logit, target_x, target_mask, target_y, target_logit):
        value = _clean(query_x, query_mask)
        _check_rows(query_logit, query_x, "query_logit")
        summaries = []
        for x, mask, y, logit in ((source_x, source_mask, source_y, source_logit),
                                  (target_x, target_mask, target_y, target_logit)):
            _clean(x, mask)
            _check_rows(y, x, "support_y")
            _check_rows(logit, x, "support_logit")
            residual = y - logit.sigmoid()
            summaries.append(torch.stack((residual.mean(1), residual.square().mean(1),
                                          mask.to(x.dtype).mean((1, 2))), -1))
        source, target = summaries
        missing = query_mask.to(query_x.dtype).mean(-1)
        repeated = [a[:, None].expand_as(query_logit) for a in
                    (*source.unbind(-1), *target.unbind(-1))]
        descriptors = torch.stack((query_logit, missing, value.mean(-1), value.square().mean(-1),
                                   *repeated, missing * repeated[0], missing * repeated[3]), -1)
        return query_logit + self.head(descriptors).squeeze(-1)
