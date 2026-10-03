"""Support-label latent refinement with a separate observed-view JEPA objective."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def _clean(x, mask):
    if x.ndim != 3 or x.shape != mask.shape or mask.dtype != torch.bool or min(x.shape) == 0:
        raise ValueError("Require nonempty [B,N,D] values and boolean missing mask")
    value = torch.where(mask, torch.zeros_like(x), x)
    if not torch.isfinite(value).all():
        raise ValueError("Observed values must be finite")
    return value


def _row_check(value, x, name):
    if value.shape != x.shape[:2] or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite [B,N]")


class _Encoder(nn.Module):
    """Shared field tokenizer; support residual moments supply field semantics."""
    def __init__(self, width=16):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(14, 2 * width), nn.GELU(), nn.Linear(2 * width, width), nn.GELU())
        self.row = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width))

    @staticmethod
    def statistics(x, mask, y, logit):
        value = torch.tanh(_clean(x, mask) / 4)
        observed = (~mask).to(x.dtype)
        count = observed.sum(1).clamp_min(1)
        residual = y - logit.sigmoid()
        mean = (value * observed).sum(1) / count
        variance = ((value - mean[:, None]).square() * observed).sum(1) / count
        return torch.stack((mean, (variance + 1e-6).sqrt(), mask.to(x.dtype).mean(1),
                            (value * observed * residual[..., None]).sum(1) / count,
                            (mask.to(x.dtype) * residual[..., None]).mean(1),
                            residual.mean(1)[:, None].expand_as(mean)), -1)

    def descriptors(self, source_x, source_mask, source_y, source_logit,
                    target_x, target_mask, target_y, target_logit):
        return torch.cat((self.statistics(source_x, source_mask, source_y, source_logit),
                          self.statistics(target_x, target_mask, target_y, target_logit)), -1)

    def rows(self, x, mask, descriptors):
        value = torch.tanh(_clean(x, mask) / 4)
        context = descriptors[:, None].expand(-1, x.shape[1], -1, -1)
        fields = self.field(torch.cat((value[..., None], mask.to(x.dtype)[..., None], context), -1))
        return self.row(torch.cat((fields.mean(2), fields.amax(2)), -1))


class RecursiveJEPA(nn.Module):
    """Three tied refinements and bounded support-gradient residual-head updates."""
    def __init__(self, width=16, steps=3, use_jepa=True, head_step=.25, ridge=.1):
        super().__init__()
        if width < 2 or steps not in (1, 3) or not 0 < head_step <= .25 or ridge <= 0:
            raise ValueError("Require width>=2, K=1 or3, bounded positive head step and positive ridge")
        self.width, self.steps, self.use_jepa = int(width), int(steps), bool(use_jepa)
        self.head_step, self.ridge = float(head_step), float(ridge)
        self.encoder = _Encoder(width)
        self.refiner = nn.Sequential(nn.Linear(7 * width + 3, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.predictor = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, decay=.99):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.mul_(decay).add_(student, alpha=1 - decay)
        for teacher, student in zip(self.teacher.buffers(), self.encoder.buffers()):
            teacher.copy_(student)
        self.teacher_updates.add_(1)

    @staticmethod
    def _validate(inputs):
        query_x, query_mask, query_logit, source_x, source_mask, source_y, source_logit, target_x, target_mask, target_y, target_logit = inputs
        for x, mask, logit in ((query_x, query_mask, query_logit),
                               (source_x, source_mask, source_logit), (target_x, target_mask, target_logit)):
            _clean(x, mask)
            _row_check(logit, x, 'frozen_logit')
            if x.shape[0] != query_x.shape[0] or x.shape[2] != query_x.shape[2]:
                raise ValueError("All sets must share batch and field dimensions")
        for x, labels in ((source_x, source_y), (target_x, target_y)):
            _row_check(labels, x, 'support_y')
            if not ((labels >= 0) & (labels <= 1)).all():
                raise ValueError("Support labels must be in [0,1]")

    @staticmethod
    def _memory(latent, labels, logit):
        positive = (latent * labels[..., None]).sum(1) / labels.sum(1).clamp_min(1)[:, None]
        negative_weights = 1 - labels
        negative = (latent * negative_weights[..., None]).sum(1) / negative_weights.sum(1).clamp_min(1)[:, None]
        residual = (latent * (labels - logit.sigmoid())[..., None]).mean(1)
        return positive, negative, residual

    def _refine(self, latent, memory, role):
        roles = latent.new_zeros(*latent.shape[:2], 3)
        roles[..., role] = 1
        expanded_memory = memory[:, None].expand(-1, latent.shape[1], -1)
        delta = .25 * self.refiner(torch.cat((latent, expanded_memory, roles), -1)).tanh()
        return (latent + delta).tanh()

    @staticmethod
    def _basis(latent):
        return torch.cat((torch.ones_like(latent[..., :1]), latent), -1)

    def _corrected(self, latent, coefficients, frozen_logit):
        raw = (self._basis(latent) * coefficients[:, None]).sum(-1)
        return frozen_logit + 2 * (raw / 2).tanh()

    def forward_details(self, query_x, query_mask, query_logit, source_x, source_mask,
                        source_y, source_logit, target_x, target_mask, target_y, target_logit,
                        force_frozen=False):
        inputs = (query_x, query_mask, query_logit, source_x, source_mask,
                  source_y, source_logit, target_x, target_mask, target_y, target_logit)
        self._validate(inputs)
        descriptors = self.encoder.descriptors(*inputs[3:])
        base = dict(source=self.encoder.rows(source_x, source_mask, descriptors),
                    target=self.encoder.rows(target_x, target_mask, descriptors),
                    query=self.encoder.rows(query_x, query_mask, descriptors))
        source, target, query = (base[key].tanh() for key in ('source', 'target', 'query'))
        coefficients = query.new_zeros(query.shape[0], self.width + 1)
        step_logits, step_latents, step_coefficients, step_support_logits = [], [], [], []
        for _ in range(self.steps):
            memory = torch.cat((*self._memory(source, source_y, self._corrected(source, coefficients, source_logit)),
                                *self._memory(target, target_y, self._corrected(target, coefficients, target_logit))), -1)
            # Update support latents together, then condition query on current supports.
            source = self._refine(source, memory, 0)
            target = self._refine(target, memory, 1)
            current_memory = torch.cat((*self._memory(source, source_y, self._corrected(source, coefficients, source_logit)),
                                        *self._memory(target, target_y, self._corrected(target, coefficients, target_logit))), -1)
            query = self._refine(query, current_memory, 2)
            basis = self._basis(target)
            raw = (basis * coefficients[:, None]).sum(-1)
            prediction = (target_logit + 2 * (raw / 2).tanh()).sigmoid()
            # Analytic derivative includes the bounded correction's Jacobian.
            error = (prediction - target_y) * (1 - (raw / 2).tanh().square())
            gradient = (basis * error[..., None]).mean(1) + self.ridge * coefficients
            coefficients = coefficients - self.head_step * gradient.tanh()
            logits = self._corrected(query, coefficients, query_logit)
            step_logits.append(logits)
            step_latents.append(dict(source=source, target=target, query=query))
            step_coefficients.append(coefficients)
            step_support_logits.append(dict(source=self._corrected(source, coefficients, source_logit),
                                            target=self._corrected(target, coefficients, target_logit)))
        logits = query_logit if force_frozen else step_logits[-1]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite recursive prediction")
        return dict(logits=logits, step_logits=step_logits, step_latents=step_latents,
                    step_coefficients=step_coefficients, coefficients=coefficients,
                    base_latents=base, descriptors=descriptors, step_support_logits=step_support_logits)

    def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                source_y, source_logit, target_x, target_mask, target_y, target_logit,
                force_frozen=False):
        return self.forward_details(query_x, query_mask, query_logit, source_x, source_mask,
                                    source_y, source_logit, target_x, target_mask, target_y,
                                    target_logit, force_frozen=force_frozen)['logits']

    @staticmethod
    def corruption_views(x, mask):
        """Rich view drops 10%; weak view drops another30% of remaining entries."""
        observed = _clean(x, mask)
        rich_mask = mask | (torch.rand_like(x) < .1)
        weak_mask = rich_mask | (torch.rand_like(x) < .3)
        return (torch.where(weak_mask, torch.zeros_like(observed), observed), weak_mask,
                torch.where(rich_mask, torch.zeros_like(observed), observed), rich_mask)

    def training_loss(self, inputs, query_y, jepa_weight=.1, variance_weight=.01,
                      covariance_weight=.001, deep_weight=.2):
        """Query labels enter supervised losses only, never embeddings or inference."""
        if any(weight < 0 for weight in (jepa_weight, variance_weight, covariance_weight, deep_weight)):
            raise ValueError("Loss weights must be nonnegative")
        self._validate(inputs)
        _row_check(query_y, inputs[0], 'query_y')
        if not ((query_y >= 0) & (query_y <= 1)).all():
            raise ValueError("Query labels must be in [0,1]")
        details = self.forward_details(*inputs)
        supervised = F.binary_cross_entropy_with_logits(details['logits'], query_y)
        deep = (torch.stack([F.binary_cross_entropy_with_logits(logit, query_y)
                             for logit in details['step_logits'][:-1]]).mean()
                if self.steps > 1 else supervised.new_zeros(()))
        jepa = variance = covariance = supervised.new_zeros(())
        monitored_student = details['base_latents']['query'].flatten(0, 1)
        if self.use_jepa and any(weight > 0 for weight in (jepa_weight, variance_weight, covariance_weight)):
            weak_x, weak_mask, rich_x, rich_mask = self.corruption_views(inputs[0], inputs[1])
            student = self.encoder.rows(weak_x, weak_mask, details['descriptors'])
            with torch.no_grad():
                teacher_descriptors = self.teacher.descriptors(*inputs[3:])
                teacher = self.teacher.rows(rich_x, rich_mask, teacher_descriptors)
            jepa = F.mse_loss(self.predictor(student), teacher)
            flattened = student.flatten(0, 1)
            monitored_student = flattened
            centered = flattened - flattened.mean(0)
            variance = F.relu(1 - (centered.square().mean(0) + 1e-4).sqrt()).mean()
            covariance_matrix = centered.T @ centered / max(1, flattened.shape[0] - 1)
            off_diagonal = covariance_matrix - torch.diag_embed(covariance_matrix.diagonal())
            covariance = off_diagonal.square().sum() / self.width
        total = supervised + deep_weight * deep + jepa_weight * jepa + variance_weight * variance + covariance_weight * covariance
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training objective")
        breakdown = {name: float(value.detach()) for name, value in
                     dict(loss=total, supervised_bce=supervised, deep_bce=deep,
                          jepa_mse=jepa, variance=variance, covariance=covariance).items()}
        standard_deviation = monitored_student.detach().std(0, unbiased=False)
        breakdown.update(student_std_mean=float(standard_deviation.mean()),
                         student_std_min=float(standard_deviation.min()))
        return total, breakdown
