"""Simplex-ridge solver for L2-normalized H&E embeddings (float64).

    min_a ||query - a @ neighbors||_2^2 + regularization * ||a||_2^2
    s.t.  a >= 0 and sum(a) = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize

DEFAULT_TRACE_SCALE = 1e-3
DEFAULT_ABSOLUTE_EPSILON = 1e-8


class LBPTSolverError(RuntimeError):
    """QP optimization or convergence error."""


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Scalar reconstruction diagnostics for one query embedding."""

    l2_error: float
    squared_error: float
    mean_squared_error: float
    relative_l2_error: float
    target_norm: float
    reconstructed_norm: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "l2_error": self.l2_error,
            "squared_error": self.squared_error,
            "mean_squared_error": self.mean_squared_error,
            "relative_l2_error": self.relative_l2_error,
            "target_norm": self.target_norm,
            "reconstructed_norm": self.reconstructed_norm,
        }


@dataclass(frozen=True)
class KKTDiagnostics:
    """Primal and first-order optimality residuals for the simplex QP."""

    constraint_violation: float
    stationarity_residual: float
    dual_feasibility_residual: float
    complementarity_residual: float
    residual: float
    common_active_gradient: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "constraint_violation": self.constraint_violation,
            "stationarity_residual": self.stationarity_residual,
            "dual_feasibility_residual": self.dual_feasibility_residual,
            "complementarity_residual": self.complementarity_residual,
            "residual": self.residual,
            "common_active_gradient": self.common_active_gradient,
        }


@dataclass(frozen=True)
class LBPTSolution:
    """LBPT weights and optimization diagnostics."""

    alpha: np.ndarray
    regularization: float
    gram_trace: float
    objective: float
    effective_n: float
    reconstruction: ReconstructionMetrics
    kkt: KKTDiagnostics
    solver_status: str
    solver_status_code: int
    solver_message: str
    solver_iterations: int

    @property
    def weights(self) -> np.ndarray:
        """Barycentric weights."""

        return self.alpha

    @property
    def lambda_value(self) -> float:
        """Ridge coefficient."""

        return self.regularization

    def to_audit_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly scalar diagnostics (the alpha vector is a list)."""

        return {
            "alpha": self.alpha.tolist(),
            "alpha_sum": float(np.sum(self.alpha, dtype=np.float64)),
            "alpha_min": float(np.min(self.alpha)),
            "alpha_max": float(np.max(self.alpha)),
            "effective_n": self.effective_n,
            "lbpt_regularization": self.regularization,
            "lbpt_gram_trace": self.gram_trace,
            "lbpt_objective": self.objective,
            "he_reconstruction_l2_error": self.reconstruction.l2_error,
            "he_reconstruction_squared_error": self.reconstruction.squared_error,
            "he_reconstruction_relative_l2_error": (
                self.reconstruction.relative_l2_error
            ),
            "qp_solver_status": self.solver_status,
            "qp_solver_status_code": self.solver_status_code,
            "qp_solver_message": self.solver_message,
            "qp_solver_iterations": self.solver_iterations,
            "qp_constraint_violation": self.kkt.constraint_violation,
            "qp_kkt_residual": self.kkt.residual,
        }


def _as_float64_array(value: Any, name: str) -> np.ndarray:
    """Convert NumPy arrays and CPU/GPU torch tensors without importing torch."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 array") from exc
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _validate_embedding_shapes(
    query_embedding: Any,
    neighbor_embeddings: Any,
) -> tuple[np.ndarray, np.ndarray]:
    query = _as_float64_array(query_embedding, "query_embedding")
    neighbors = _as_float64_array(neighbor_embeddings, "neighbor_embeddings")

    if query.ndim != 1:
        raise ValueError(f"query_embedding must have shape (D,), got {query.shape}")
    if neighbors.ndim != 2:
        raise ValueError(
            f"neighbor_embeddings must have shape (K, D), got {neighbors.shape}"
        )
    if query.size == 0:
        raise ValueError("embedding dimension D must be positive")
    if neighbors.shape[0] == 0:
        raise ValueError("at least one selected neighbor is required")
    if neighbors.shape[1] != query.shape[0]:
        raise ValueError(
            "query and neighbor embedding dimensions differ: "
            f"{query.shape[0]} != {neighbors.shape[1]}"
        )
    if not np.all(np.isfinite(query)):
        raise ValueError("query_embedding contains NaN or infinity")
    if not np.all(np.isfinite(neighbors)):
        raise ValueError("neighbor_embeddings contains NaN or infinity")

    return query, neighbors


def _validate_normalized_embeddings(
    query: np.ndarray,
    neighbors: np.ndarray,
    normalization_tol: float,
) -> None:
    if not np.isfinite(normalization_tol) or normalization_tol <= 0.0:
        raise ValueError("normalization_tol must be finite and positive")

    query_norm = float(np.linalg.norm(query))
    neighbor_norms = np.linalg.norm(neighbors, axis=1)
    if abs(query_norm - 1.0) > normalization_tol:
        raise ValueError(
            "query_embedding must already be L2-normalized; "
            f"observed norm={query_norm:.17g}"
        )

    bad = np.flatnonzero(np.abs(neighbor_norms - 1.0) > normalization_tol)
    if bad.size:
        details = ", ".join(
            f"{int(index)}:{float(neighbor_norms[index]):.17g}" for index in bad[:5]
        )
        raise ValueError(
            "neighbor_embeddings must already be row-wise L2-normalized; "
            f"bad row norms ({details})"
        )


def _validate_simplex(
    weights: Any,
    expected_size: int,
    constraint_tol: float,
) -> tuple[np.ndarray, float]:
    alpha = _as_float64_array(weights, "weights")
    if alpha.ndim != 1 or alpha.shape[0] != expected_size:
        raise ValueError(
            f"weights must have shape ({expected_size},), got {alpha.shape}"
        )
    if not np.all(np.isfinite(alpha)):
        raise ValueError("weights contain NaN or infinity")
    if not np.isfinite(constraint_tol) or constraint_tol <= 0.0:
        raise ValueError("constraint_tol must be finite and positive")

    violation = max(
        abs(float(np.sum(alpha, dtype=np.float64)) - 1.0),
        max(0.0, -float(np.min(alpha))),
        max(0.0, float(np.max(alpha)) - 1.0),
    )
    if violation > constraint_tol:
        raise ValueError(
            "weights violate the probability simplex: "
            f"maximum violation={violation:.3e}, tolerance={constraint_tol:.3e}"
        )
    return alpha, float(violation)


def effective_neighbor_number(
    weights: Any,
    *,
    constraint_tol: float = 1e-9,
) -> float:
    """Return the effective neighbor count, ``1 / sum(alpha**2)``."""

    raw = _as_float64_array(weights, "weights")
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    alpha, _ = _validate_simplex(raw, raw.size, constraint_tol)
    denominator = float(np.dot(alpha, alpha))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("weights have an invalid squared norm")
    return float(1.0 / denominator)


def compute_reconstruction_metrics(
    query_embedding: Any,
    neighbor_embeddings: Any,
    weights: Any,
    *,
    constraint_tol: float = 1e-9,
    relative_floor: float = 1e-12,
) -> ReconstructionMetrics:
    """Compute reconstruction errors in the supplied embedding space."""

    query, neighbors = _validate_embedding_shapes(query_embedding, neighbor_embeddings)
    alpha, _ = _validate_simplex(weights, neighbors.shape[0], constraint_tol)
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")

    reconstructed = alpha @ neighbors
    residual = query - reconstructed
    squared_error = float(np.dot(residual, residual))
    l2_error = float(np.sqrt(max(0.0, squared_error)))
    target_norm = float(np.linalg.norm(query))
    reconstructed_norm = float(np.linalg.norm(reconstructed))

    return ReconstructionMetrics(
        l2_error=l2_error,
        squared_error=squared_error,
        mean_squared_error=float(squared_error / query.size),
        relative_l2_error=float(l2_error / max(target_norm, relative_floor)),
        target_norm=target_norm,
        reconstructed_norm=reconstructed_norm,
    )


def _objective_and_gradient(
    alpha: np.ndarray,
    query: np.ndarray,
    neighbors: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    residual = alpha @ neighbors - query
    objective = float(
        np.dot(residual, residual) + regularization * np.dot(alpha, alpha)
    )
    gradient = 2.0 * (neighbors @ residual + regularization * alpha)
    return objective, np.asarray(gradient, dtype=np.float64)


def _compute_kkt_diagnostics(
    alpha: np.ndarray,
    gradient: np.ndarray,
    constraint_violation: float,
    active_tol: float,
) -> KKTDiagnostics:
    active = alpha > active_tol
    if not np.any(active):
        raise LBPTSolverError("QP result has no active simplex coordinate")

    common_gradient = float(np.mean(gradient[active], dtype=np.float64))
    stationarity_residual = float(
        np.max(np.abs(gradient[active] - common_gradient), initial=0.0)
    )

    inactive = ~active
    dual_feasibility_residual = float(
        np.max(
            np.maximum(common_gradient - gradient[inactive], 0.0),
            initial=0.0,
        )
    )
    reduced_gradient = gradient - common_gradient
    complementarity_residual = float(
        np.max(np.abs(alpha * reduced_gradient), initial=0.0)
    )
    residual = max(
        constraint_violation,
        stationarity_residual,
        dual_feasibility_residual,
        complementarity_residual,
    )

    return KKTDiagnostics(
        constraint_violation=float(constraint_violation),
        stationarity_residual=stationarity_residual,
        dual_feasibility_residual=dual_feasibility_residual,
        complementarity_residual=complementarity_residual,
        residual=float(residual),
        common_active_gradient=common_gradient,
    )


def solve_lbpt_simplex(
    query_embedding: Any,
    neighbor_embeddings: Any,
    *,
    trace_scale: float = DEFAULT_TRACE_SCALE,
    eps: float = DEFAULT_ABSOLUTE_EPSILON,
    normalization_tol: float = 1e-6,
    constraint_tol: float = 1e-9,
    kkt_tol: float = 1e-7,
    active_tol: float = 1e-8,
    optimizer_ftol: float = 1e-15,
    maxiter: int = 1000,
) -> LBPTSolution:
    """Solve and verify the H&E-LBPT simplex-ridge QP.

    ``G`` is the local Gram matrix of row-wise differences
    ``neighbor_embeddings - query_embedding``.  The ridge coefficient is
    exactly ``trace_scale * trace(G) + eps``.
    """

    query, neighbors = _validate_embedding_shapes(query_embedding, neighbor_embeddings)
    _validate_normalized_embeddings(query, neighbors, normalization_tol)

    for name, value, allow_zero in (
        ("trace_scale", trace_scale, True),
        ("eps", eps, False),
        ("kkt_tol", kkt_tol, False),
        ("active_tol", active_tol, False),
        ("optimizer_ftol", optimizer_ftol, False),
    ):
        if not np.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise ValueError("maxiter must be a positive integer")
    if maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")

    differences = neighbors - query[None, :]
    local_gram = differences @ differences.T
    gram_trace = float(np.trace(local_gram, dtype=np.float64))
    regularization = float(trace_scale * gram_trace + eps)
    if not np.isfinite(gram_trace) or not np.isfinite(regularization):
        raise ValueError("local Gram trace or regularization is not finite")
    if regularization <= 0.0:
        raise ValueError("regularization must be strictly positive")

    k = neighbors.shape[0]
    if k == 1:
        alpha = np.ones(1, dtype=np.float64)
        status = "analytic_k1"
        status_code = 0
        message = "K=1 has a unique feasible simplex coordinate"
        iterations = 0
    else:
        initial = np.full(k, 1.0 / k, dtype=np.float64)

        def objective(alpha_value: np.ndarray) -> float:
            return _objective_and_gradient(
                alpha_value, query, neighbors, regularization
            )[0]

        def gradient(alpha_value: np.ndarray) -> np.ndarray:
            return _objective_and_gradient(
                alpha_value, query, neighbors, regularization
            )[1]

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=gradient,
            bounds=[(0.0, 1.0)] * k,
            constraints=(
                {
                    "type": "eq",
                    "fun": lambda alpha_value: float(np.sum(alpha_value) - 1.0),
                    "jac": lambda alpha_value: np.ones_like(
                        alpha_value, dtype=np.float64
                    ),
                },
            ),
            options={
                "ftol": float(optimizer_ftol),
                "maxiter": int(maxiter),
                "disp": False,
            },
        )
        if not bool(result.success):
            raise LBPTSolverError(
                "SLSQP failed for the LBPT simplex QP: "
                f"status={getattr(result, 'status', 'unknown')}, "
                f"message={getattr(result, 'message', 'unknown')}"
            )

        alpha = _as_float64_array(result.x, "optimizer result")
        status = "slsqp_success"
        status_code = int(getattr(result, "status", 0))
        message = str(getattr(result, "message", "Optimization succeeded"))
        iterations = int(getattr(result, "nit", -1))

    try:
        alpha, constraint_violation = _validate_simplex(alpha, k, constraint_tol)
    except ValueError as exc:
        raise LBPTSolverError(f"invalid QP solution: {exc}") from exc

    objective_value, objective_gradient = _objective_and_gradient(
        alpha, query, neighbors, regularization
    )
    if not np.isfinite(objective_value) or not np.all(np.isfinite(objective_gradient)):
        raise LBPTSolverError("QP objective or gradient is not finite")

    kkt = _compute_kkt_diagnostics(
        alpha,
        objective_gradient,
        constraint_violation,
        active_tol,
    )
    if kkt.residual > kkt_tol:
        raise LBPTSolverError(
            "SLSQP reported success but the LBPT solution failed KKT checks: "
            f"residual={kkt.residual:.3e}, tolerance={kkt_tol:.3e}, "
            f"stationarity={kkt.stationarity_residual:.3e}, "
            f"dual_feasibility={kkt.dual_feasibility_residual:.3e}, "
            f"complementarity={kkt.complementarity_residual:.3e}"
        )

    reconstruction = compute_reconstruction_metrics(
        query,
        neighbors,
        alpha,
        constraint_tol=constraint_tol,
    )
    effective_n = effective_neighbor_number(alpha, constraint_tol=constraint_tol)

    # Store immutable weights for caching.
    alpha.setflags(write=False)
    return LBPTSolution(
        alpha=alpha,
        regularization=regularization,
        gram_trace=gram_trace,
        objective=objective_value,
        effective_n=effective_n,
        reconstruction=reconstruction,
        kkt=kkt,
        solver_status=status,
        solver_status_code=status_code,
        solver_message=message,
        solver_iterations=iterations,
    )


solve_lbpt_simplex_ridge = solve_lbpt_simplex


__all__ = [
    "DEFAULT_ABSOLUTE_EPSILON",
    "DEFAULT_TRACE_SCALE",
    "KKTDiagnostics",
    "LBPTSolution",
    "LBPTSolverError",
    "ReconstructionMetrics",
    "compute_reconstruction_metrics",
    "effective_neighbor_number",
    "solve_lbpt_simplex",
    "solve_lbpt_simplex_ridge",
]
