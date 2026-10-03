"""Learn support regularization through a shared, differentiable logistic solver."""
import math

import torch
from torch import nn
from torch.nn import functional as F


MIN_PENALTY = .01
MAX_PENALTY = 1.
INITIAL_PENALTY = .1
INITIAL_RAW = math.log((INITIAL_PENALTY - MIN_PENALTY) / (MAX_PENALTY - INITIAL_PENALTY))


def _values(x, mask):
    if x.ndim != 3 or x.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Expected [B,N,D] values and equally shaped boolean missing masks")
    if min(x.shape) == 0:
        raise ValueError("Batch, rows and fields must be nonempty")
    values = torch.where(mask, torch.zeros_like(x), x)
    if not torch.isfinite(values).all():
        raise ValueError("Observed values must be finite")
    return values


def _rows(value, x, name):
    if value.shape != x.shape[:2] or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and shaped [B,N]")


def feature_basis(x, mask, logit, dtype=torch.float64):
    """Intercept, frozen logit, values, masks, all ordered value-mask pairs."""
    value = _values(x, mask).to(dtype)
    _rows(logit, x, "logit")
    missing = mask.to(dtype)
    interaction = (value[..., :, None] * missing[..., None, :]).flatten(-2)
    return torch.cat((torch.ones_like(logit, dtype=dtype)[..., None],
                      logit.to(dtype)[..., None], value, missing, interaction), -1)


basis = feature_basis


def coordinate_types(columns, device, dtype):
    """The type follows the coordinate under any joint column permutation."""
    sizes = (1, 1, columns, columns, columns * columns)
    return torch.cat([F.one_hot(torch.full((size,), index, device=device), 5)
                      for index, size in enumerate(sizes)]).to(dtype)


def support_statistics(basis, labels, logit):
    """Coordinate moments and label residual associations, using support only."""
    residual = labels.to(basis.dtype) - logit.to(basis.dtype).sigmoid()
    centered = basis - basis.mean(1, keepdim=True)
    product = basis * residual[..., None]
    return torch.stack((basis.mean(1), basis.square().mean(1), centered.square().mean(1),
                        product.mean(1), product.square().mean(1),
                        residual.mean(1)[:, None].expand(-1, basis.shape[-1])), -1)


def penalty_from_raw(raw):
    return MIN_PENALTY + (MAX_PENALTY - MIN_PENALTY) * raw.sigmoid()


class _RidgeSolver(nn.Module):
    """All controls share this basis, offset, objective and four Newton updates."""
    def __init__(self, steps=4, solver_dtype=torch.float64):
        super().__init__()
        if steps != 4:
            raise ValueError("This version commits exactly four Newton updates")
        if solver_dtype not in (torch.float32, torch.float64):
            raise ValueError("Solver dtype must be float32 or float64")
        self.steps = int(steps)
        self.solver_dtype = solver_dtype

    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        raise NotImplementedError

    def solve(self, basis, labels, offset, penalties):
        """Minimize mean BCE(offset + basis @ theta) + .5 sum(lambda theta²)."""
        labels, offset = labels.to(self.solver_dtype), offset.to(self.solver_dtype)
        coefficients = torch.zeros_like(penalties, dtype=self.solver_dtype)
        penalties = penalties.to(self.solver_dtype)
        count = basis.shape[1]
        for _ in range(self.steps):
            prediction = (offset + (basis * coefficients[:, None, :]).sum(-1)).sigmoid()
            gradient = (basis * (prediction - labels)[..., None]).mean(1) + penalties * coefficients
            weight = prediction * (1 - prediction)
            hessian = basis.transpose(1, 2) @ (basis * weight[..., None]) / count
            hessian = hessian + torch.diag_embed(penalties)
            step = torch.linalg.solve(hessian, gradient[..., None]).squeeze(-1)
            coefficients = coefficients - step
        return coefficients

    def fit_support(self, source_x, source_mask, source_y, source_logit,
                    target_x, target_mask, target_y, target_logit):
        if source_x.shape[0] != target_x.shape[0] or source_x.shape[2] != target_x.shape[2]:
            raise ValueError("Source and target must share batch and field dimensions")
        for x, y in ((source_x, source_y), (target_x, target_y)):
            _rows(y, x, "support_y")
            if not ((y >= 0) & (y <= 1)).all():
                raise ValueError("Support labels must be in [0,1]")
        source_basis = feature_basis(source_x, source_mask, source_logit, self.solver_dtype)
        target_basis = feature_basis(target_x, target_mask, target_logit, self.solver_dtype)
        penalties = self.penalties(source_basis, source_y, source_logit,
                                   target_basis, target_y, target_logit, source_x.shape[2])
        coefficients = self.solve(target_basis, target_y, target_logit, penalties)
        return coefficients, penalties

    def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                source_y, source_logit, target_x, target_mask, target_y, target_logit):
        if query_x.shape[0] != target_x.shape[0] or query_x.shape[2] != target_x.shape[2]:
            raise ValueError("Query and target must share batch and field dimensions")
        coefficients, _ = self.fit_support(source_x, source_mask, source_y, source_logit,
                                            target_x, target_mask, target_y, target_logit)
        basis = feature_basis(query_x, query_mask, query_logit, self.solver_dtype)
        correction = (basis * coefficients[:, None, :]).sum(-1)
        output = query_logit + correction.to(query_logit.dtype)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite solver prediction")
        return output

    @torch.no_grad()
    def solver_audit(self, query_x, query_mask, query_logit, source_x, source_mask,
                     source_y, source_logit, target_x, target_mask, target_y, target_logit):
        coefficients, penalties = self.fit_support(source_x, source_mask, source_y, source_logit,
                                                    target_x, target_mask, target_y, target_logit)
        basis = feature_basis(target_x, target_mask, target_logit, self.solver_dtype)
        penalties = penalties.to(self.solver_dtype)
        logits = target_logit.to(self.solver_dtype) + (basis * coefficients[:, None, :]).sum(-1)
        gradient = (basis * (logits.sigmoid() - target_y.to(self.solver_dtype))[..., None]).mean(1)
        gradient = gradient + penalties * coefficients
        objective = F.binary_cross_entropy_with_logits(logits, target_y.to(self.solver_dtype), reduction='none').mean(1)
        objective = objective + .5 * (penalties * coefficients.square()).sum(1)
        return dict(newton_steps=self.steps, solver_dtype=str(self.solver_dtype),
                    max_abs_gradient=float(gradient.abs().max()),
                    penalty_min=float(penalties.min()), penalty_max=float(penalties.max()),
                    coefficient_max_abs=float(coefficients.abs().max()),
                    objective_mean=float(objective.mean()),
                    all_finite=bool(torch.isfinite(coefficients).all() & torch.isfinite(objective).all()))

    def diagnostics(self, *inputs):
        """Detached JSON-ready support solver audit, never a convergence claim."""
        return self.solver_audit(*inputs)


class SupportRegularizer(_RidgeSolver):
    """Shared coordinate MLP learns penalties from source/target support moments."""
    def __init__(self, width=32, steps=4, solver_dtype=torch.float64):
        super().__init__(steps, solver_dtype)
        self.network = nn.Sequential(nn.Linear(23, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, INITIAL_RAW)

    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        source = support_statistics(source_basis, source_y, source_logit)
        target = support_statistics(target_basis, target_y, target_logit)
        types = coordinate_types(columns, source_basis.device, source_basis.dtype)
        types = types[None].expand(source_basis.shape[0], -1, -1)
        # Compress descriptor magnitude while retaining signed associations.
        moments = torch.cat((source, target, target - source), -1)
        moments = torch.sign(moments) * torch.log1p(moments.abs())
        descriptors = torch.cat((moments, types), -1).to(self.network[0].weight.dtype)
        return penalty_from_raw(self.network(descriptors).squeeze(-1))


class FeatureTypeRegularizer(SupportRegularizer):
    """Identical MLP size, with support statistics excluded from its input."""
    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        types = coordinate_types(columns, source_basis.device, self.network[0].weight.dtype)
        types = types[None].expand(source_basis.shape[0], -1, -1)
        descriptors = torch.cat((types.new_zeros(*types.shape[:-1], 18), types), -1)
        return penalty_from_raw(self.network(descriptors).squeeze(-1))


class TargetOnlyRegularizer(SupportRegularizer):
    """Matched MLP; source slots duplicate target moments, difference slots are zero."""
    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        return super().penalties(target_basis, target_y, target_logit,
                                 target_basis, target_y, target_logit, columns)


class GlobalRegularizer(_RidgeSolver):
    """A single learned penalty, shared across every task and basis coordinate."""
    def __init__(self, steps=4, solver_dtype=torch.float64):
        super().__init__(steps, solver_dtype)
        self.raw_penalty = nn.Parameter(torch.tensor(INITIAL_RAW))

    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        return penalty_from_raw(self.raw_penalty).expand(source_basis.shape[0], source_basis.shape[-1])


class FixedRegularizer(_RidgeSolver):
    """Fixed-penalty control with exactly the same four-step solver objective."""
    def __init__(self, penalty=.1, steps=4, solver_dtype=torch.float64):
        super().__init__(steps, solver_dtype)
        if not MIN_PENALTY <= penalty <= MAX_PENALTY:
            raise ValueError("Fixed penalty must be in [.01,1]")
        self.register_buffer('penalty', torch.tensor(float(penalty)))

    def penalties(self, source_basis, source_y, source_logit,
                  target_basis, target_y, target_logit, columns):
        return self.penalty.expand(source_basis.shape[0], source_basis.shape[-1])
