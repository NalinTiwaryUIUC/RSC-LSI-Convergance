from .grad_norms import compute_grad_norm_sq
from .probes import evaluate_probes, get_probe_value_for_grad
from .random_projections import get_or_create_logit_projection, get_or_create_param_projections

# Probes for which we compute grad norms (proxy LSI)
PROBES_FOR_GRAD_NORM = ("f_nll", "f_margin", "f_pc1", "f_proj1", "f_dist")

__all__ = [
    "evaluate_probes",
    "get_probe_value_for_grad",
    "compute_grad_norm_sq",
    "get_or_create_param_projections",
    "get_or_create_logit_projection",
    "PROBES_FOR_GRAD_NORM",
]
