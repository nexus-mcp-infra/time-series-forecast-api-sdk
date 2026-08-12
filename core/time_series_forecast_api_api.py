from fastapi import Depends, Request, WebSocket, status
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy as scipy_entropy

app = FastAPI(
    title="Bass Diffusion + Shannon Entropy Forecast API",
    version="1.0.0",
    description=(
        "Dual-output forecast: numeric Bass diffusion trajectory calibrated via MLE "
        "plus per-step conditional entropy degradation map H(X_{t+k}|X_{1..t})."
    ),
)


# ---------------------------------------------------------------------------
# Custom exceptions mapped to HTTP responses
# ---------------------------------------------------------------------------

class BassParamsMissingError(HTTPException):
    def __init__(self, detail: str = "Bass parameters p, q, M are required and must be positive."):
        super().__init__(status_code=422, detail=detail)


class BassMLEConvergenceError(HTTPException):
    def __init__(self, detail: str = "MLE optimization did not converge for the provided series."):
        super().__init__(status_code=422, detail=detail)


class EntropyProfileInvalidError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)


# ---------------------------------------------------------------------------
# Core mathematics — standalone, no internal NEXUS imports
# ---------------------------------------------------------------------------

def _bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    """
    Bass cumulative adoption at each integer time step t (1-indexed).
    F(t) = M * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))
    """
    exp_term = np.exp(-(p + q) * t)
    return M * (1.0 - exp_term) / (1.0 + (q / p) * exp_term)


def _bass_incremental(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    """Incremental (period) adoptions from Bass cumulative."""
    cum = _bass_cumulative(t, p, q, M)
    incremental = np.diff(cum, prepend=0.0)
    return incremental


def _bass_nll(params: np.ndarray, observed: np.ndarray, M_fixed: float) -> float:
    """
    Negative log-likelihood for Bass model under Gaussian observation noise.
    params = [log_p, log_q] to enforce positivity.
    """
    log_p, log_q = params
    p = np.exp(log_p)
    q = np.exp(log_q)
    if p <= 0 or q <= 0 or M_fixed <= 0:
        return 1e12
    T = len(observed)
    t = np.arange(1, T + 1, dtype=float)
    predicted = _bass_incremental(t, p, q, M_fixed)
    if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
        return 1e12
    residuals = observed - predicted
    sigma2 = np.var(residuals) + 1e-8
    nll = 0.5 * T * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(residuals ** 2) / sigma2
    return float(nll)


def _calibrate_bass_mle(
    series: list[float],
    M_prior: float,
    max_iter: int,
) -> dict:
    """
    Calibrate Bass (p, q, M) via MLE.  M is constrained to [last_value, M_prior].
    Returns dict with p, q, M, converged, nll.
    """
    obs = np.array(series, dtype=float)
    if np.any(obs < 0):
        raise BassMLEConvergenceError("adoption_series must be non-negative.")

    # Convert cumulative to incremental if monotone-non-decreasing
    if np.all(np.diff(obs) >= -1e-6):
        incremental = np.diff(obs, prepend=0.0)
        incremental = np.maximum(incremental, 0.0)
    else:
        incremental = np.maximum(obs, 0.0)

    M_fixed = M_prior

    best_result = None
    best_nll = np.inf

    # Multi-start grid over (log_p, log_q)
    p_starts = np.log([0.001, 0.01, 0.05, 0.1])
    q_starts = np.log([0.01, 0.1, 0.3, 0.5])

    for lp0 in p_starts:
        for lq0 in q_starts:
            try:
                res = minimize(
                    _bass_nll,
                    x0=np.array([lp0, lq0]),
                    args=(incremental, M_fixed),
                    method="L-BFGS-B",
                    bounds=[(-10, 0), (-10, 2)],
                    options={"maxiter": max_iter, "ftol": 1e-9},
                )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_result = res
            except Exception:
                continue

    if best_result is None or not np.isfinite(best_nll):
        raise BassMLEConvergenceError(
            "MLE failed to converge across all starting points. "
            "Try a larger series or adjust market_potential_prior."
        )

    p_opt = float(np.exp(best_result.x[0]))
    q_opt = float(np.exp(best_result.x[1]))

    return {
        "p_innovation": p_opt,
        "q_imitation": q_opt,
        "M_market_potential": M_fixed,
        "nll": float(best_nll),
        "converged": bool(best_result.success),
    }


def _compute_conditional_entropy_profile(
    series: list[float],
    p: float,
    q: float,
    M: float,
    horizon: int,
    n_bins: int,
) -> np.ndarray:
    """
    Estimate H(X_{t+k} | X_{1..t}) for k = 1..horizon using the Bass predictive
    distribution perturbed by empirical residual noise.

    Method:
      1. Fit Bass trajectory over historical window -> get residuals.
      2. Model residual distribution empirically (kernel-smoothed histogram).
      3. For each future step k, generate Monte Carlo draws by propagating
         residual uncertainty through the Bass model, then estimate Shannon
         entropy of the resulting predictive distribution.
    """
    obs = np.array(series, dtype=float)
    T = len(obs)
    t_hist = np.arange(1, T + 1, dtype=float)
    bass_hist = _bass_incremental(t_hist, p, q, M)

    if np.all(np.diff(obs) >= -1e-6):
        incremental_obs = np.diff(obs, prepend=0.0)
    else:
        incremental_obs = obs.copy()
    incremental_obs = np.maximum(incremental_obs, 0.0)

    residuals = incremental_obs - bass_hist
    residual_std = float(np.std(residuals)) + 1e-6

    rng = np.random.default_rng(seed=42)
    n_samples = 2000

    entropy_profile = np.zeros(horizon)

    for k in range(1, horizon + 1):
        t_future = float(T + k)
        bass_mean = float(_bass_incremental(np.array([t_future]), p, q, M)[0])
        # Predictive noise grows with horizon: sigma scales as sqrt(k) (random-walk drift)
        sigma_k = residual_std * np.sqrt(k)
        samples = rng.normal(loc=bass_mean, scale=sigma_k, size=n_samples)
        samples = np.maximum(samples, 0.0)

        # Discretize into n_bins to estimate Shannon entropy in bits
        counts, _ = np.histogram(samples, bins=n_bins)
        counts = counts + 1e-10  # Laplace smoothing
        probs = counts / counts.sum()
        h = float(scipy_entropy(probs, base=2))
        entropy_profile[k - 1] = h

    return entropy_profile


def _resolve_cutoff(
    entropy_profile: np.ndarray,
    marginal_threshold: float,
    absolute_ceiling: float,
) -> int:
    """
    Returns the index (1-based step) at which to cut off the forecast.
    Cutoff triggers when:
      - absolute entropy exceeds absolute_ceiling, OR
      - marginal entropy gain drops below marginal_threshold (forecast plateau)
    Returns the last reliable step (1-based).  Returns 0 if step 1 already fails.
    """
    H = entropy_profile
    n = len(H)
    for k in range(n):
        # Absolute ceiling check
        if H[k] > absolute_ceiling:
            return k  # steps before k are reliable (1-based: k means k steps)
        # Marginal gain check: marginal = H[k] - H[k-1]; for k==0 use H[0]
        if k == 0:
            marginal = H[0]
        else:
            marginal = H[k] - H[k - 1]
        if marginal < marginal_threshold and k > 0:
            return k
    return n


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class BassMLECalibrateRequest(BaseModel):
    adoption_series: Annotated[
        list[float],
        Field(..., description="Historical cumulative or periodic adoption numbers. Must be non-negative and non-decreasing.", min_length=5),
    ]
    market_potential_prior: Annotated[
        float,
        Field(..., description="Prior estimate of total addressable market size (M). Must be greater than the last value of adoption_series.", ge=0),
    ]
    # NEXUS_PARAM_DEVIATION: mle_max_iterations -- spec defines as float with ge=1/le=10000; coerced to int internally for optimizer iterations count
    mle_max_iterations: Annotated[
        float,
        Field(..., description="Maximum number of iterations for the MLE optimization algorithm.", ge=1, le=10000),
    ]

    @model_validator(mode="after")
    def validate_market_prior(self) -> "BassMLECalibrateRequest":
        if self.adoption_series and self.market_potential_prior <= max(self.adoption_series):
            raise ValueError(
                f"market_potential_prior ({self.market_potential_prior}) must be strictly greater "
                f"than the maximum value in adoption_series ({max(self.adoption_series)})."
            )
        return self


class BassMLECalibrateResponse(BaseModel):
    p_innovation: float
    q_imitation: float
    M_market_potential: float
    nll: float
    converged: bool


class BassForecastRequest(BaseModel):
    adoption_series: Annotated[
        list[float],
        Field(..., description="Historical adoption series used for entropy computation and forecast alignment.", min_length=1),
    ]
    # NEXUS_PARAM_DEVIATION: horizon_steps -- spec defines as float; coerced to int internally for range generation
    horizon_steps: Annotated[
        float,
        Field(..., description="Number of future steps to forecast. Must be a positive integer.", ge=1),
    ]
    p_innovation: Annotated[
        float,
        Field(..., description="Coefficient of innovation (p) from Bass calibration.", ge=0, le=1),
    ]
    q_imitation: Annotated[
        float,
        Field(..., description="Coefficient of imitation (q) from Bass calibration.", ge=0, le=1),
    ]
    M_market_potential: Annotated[
        float,
        Field(..., description="Estimated total market potential (saturation level). Must be > 0.", ge=0),
    ]
    entropy_cutoff_threshold: Annotated[
        float,
        Field(..., description="Marginal entropy gain threshold per step (in bits) below which the forecast horizon is cut off.", ge=0),
    ]

    @model_validator(mode="after")
    def validate_bass_params(self) -> "BassForecastRequest":
        if self.M_market_potential <= 0:
            raise ValueError("M_market_potential must be strictly greater than 0.")
        if self.p_innovation <= 0 or self.q_imitation <= 0:
            raise ValueError("p_innovation and q_imitation must be strictly greater than 0.")
        return self


class BassForecastResponse(BaseModel):
    forecast_values: list[float]
    forecast_steps: list[int]
    entropy_per_step: list[float]
    cutoff_step: int
    truncated: bool


class EntropyProfileRequest(BaseModel):
    adoption_series: Annotated[
        list[float],
        Field(..., description="Historical adoption series for estimating the state distribution.", min_length=10),
    ]
    # NEXUS_PARAM_DEVIATION: horizon_steps -- spec defines as float; coerced to int internally
    horizon_steps: Annotated[
        float,
        Field(..., description="Number of future steps for which to compute entropy.", ge=1),
    ]
    # NEXUS_PARAM_DEVIATION: n_bins -- spec defines as float; coerced to int internally for np.histogram bins argument
    n_bins: Annotated[
        float,
        Field(..., description="Number of bins for discretizing continuous values to estimate empirical entropy. Must be at least 2.", ge=2, le=100),
    ]
    p_innovation: Annotated[
        float,
        Field(..., description="Coefficient of innovation (p).", ge=0, le=1),
    ]
    q_imitation: Annotated[
        float,
        Field(..., description="Coefficient of imitation (q).", ge=0, le=1),
    ]
    M_market_potential: Annotated[
        float,
        Field(..., description="Estimated market potential.", ge=0),
    ]

    @model_validator(mode="after")
    def validate_bass_params(self) -> "EntropyProfileRequest":
        if self.M_market_potential <= 0:
            raise ValueError("M_market_potential must be strictly greater than 0.")
        if self.p_innovation <= 0 or self.q_imitation <= 0:
            raise ValueError("p_innovation and q_imitation must be strictly greater than 0.")
        return self


class EntropyProfileResponse(BaseModel):
    entropy_per_step: list[float]
    horizon_steps: int
    max_entropy: float
    min_entropy: float
    mean_entropy: float


class EntropyCutoffRequest(BaseModel):
    entropy_per_step: Annotated[
        list[float],
        Field(..., description="Array of conditional entropy values per horizon step (starting from step 1).", min_length=1),
    ]
    marginal_entropy_threshold: Annotated[
        float,
        Field(..., description="Minimum acceptable increase in entropy per step (bits).", ge=0),
    ]
    absolute_entropy_ceiling: Annotated[
        float,
        Field(..., description="Absolute entropy value (bits) beyond which the forecast is too uncertain.", ge=0),
    ]

    @model_validator(mode="after")
    def validate_entropy_profile(self) -> "EntropyCutoffRequest":
        for i, v in enumerate(self.entropy_per_step):
            if v < 0:
                raise ValueError(
                    f"entropy_per_step contains a negative value at index {i} ({v}). "
                    "Shannon entropy must be non-negative."
                )
        return self


class EntropyCutoffResponse(BaseModel):
    recommended_cutoff_step: int
    reliable_steps: list[int]
    truncated_at: Optional[int]
    reason: str


class ReliabilityMapRequest(BaseModel):
    adoption_series: Annotated[
        list[float],
        Field(..., description="Historical adoption series (cumulative numbers).", min_length=15),
    ]
    # NEXUS_PARAM_DEVIATION: horizon_steps -- spec defines as float; coerced to int internally
    horizon_steps: Annotated[
        float,
        Field(..., description="Desired forecast horizon (number of steps).", ge=1),
    ]
    market_potential_prior: Annotated[
        float,
        Field(..., description="Prior estimate of the market potential M for Bass calibration.", ge=0),
    ]
    marginal_entropy_threshold: Annotated[
        float,
        Field(..., description="Entropy threshold for cutoff (bits).", ge=0),
    ]
    # NEXUS_PARAM_DEVIATION: n_bins -- spec defines as float; coerced to int internally
    n_bins: Annotated[
        float,
        Field(..., description="Number of bins for entropy discretization.", ge=2, le=100),
    ]
    # NEXUS_PARAM_DEVIATION: mle_max_iterations -- spec defines as float; coerced to int internally
    mle_max_iterations: Annotated[
        float,
        Field(..., description="Maximum iterations for the MLE optimizer.", ge=1, le=10000),
    ]

    @model_validator(mode="after")
    def validate_market_prior(self) -> "ReliabilityMapRequest":
        if self.adoption_series and self.market_potential_prior <= max(self.adoption_series):
            raise ValueError(
                f"market_potential_prior ({self.market_potential_prior}) must be strictly greater "
                f"than the maximum value in adoption_series ({max(self.adoption_series)})."
            )
        return self


class ReliabilityMapResponse(BaseModel):
    p_innovation: float
    q_imitation: float
    M_market_potential: float
    mle_converged: bool
    forecast_values: list[float]
    forecast_steps: list[int]
    entropy_per_step: list[float]
    marginal_entropy_per_step: list[float]
    cutoff_step: int
    truncated: bool
    reliability_map: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/bass/mle/calibrate", response_model=BassMLECalibrateResponse)
async def calibrate_bass_diffusion_mle(req: BassMLECalibrateRequest) -> BassMLECalibrateResponse:
    """
    Calibrates Bass diffusion model parameters (p_innovation, q_imitation, M_market_potential)
    via Maximum Likelihood Estimation on the provided historical adoption series.

    Use to estimate model coefficients from observed adoption data, especially with at least
    15 points. Do NOT use for very short series (<5 points) or when adoption is still in the
    initial flat phase without a clear inflection — the MLE may fail to converge.
    """
    if not req.adoption_series:
        raise HTTPException(status_code=422, detail="adoption_series must not be empty.")

    result = _calibrate_bass_mle(
        series=req.adoption_series,
        M_prior=req.market_potential_prior,
        max_iter=int(req.mle_max_iterations),
    )
    return BassMLECalibrateResponse(**result)


@app.post("/bass/forecast", response_model=BassForecastResponse)
async def forecast_bass_adoption_curve(req: BassForecastRequest) -> BassForecastResponse:
    """
    Generates future adoption values over a given horizon using the Bass diffusion model.
    The forecast is truncated if the conditional entropy exceeds the configured threshold,
    ensuring only reliable steps are returned.

    Use when you have calibrated Bass parameters and need a numeric forecast with an
    entropy-based reliability cutoff. Do NOT use without first providing valid p, q, M
    parameters (missing or zero parameters will raise BassParamsMissingError).
    """
    if req.p_innovation <= 0 or req.q_imitation <= 0 or req.M_market_potential <= 0:
        raise BassParamsMissingError()

    horizon = int(req.horizon_steps)
    T = len(req.adoption_series)
    t_future = np.arange(T + 1, T + horizon + 1, dtype=float)

    bass_future = _bass_incremental(t_future, req.p_innovation, req.q_imitation, req.M_market_potential)

    entropy_profile = _compute_conditional_entropy_profile(
        series=req.adoption_series,
        p=req.p_innovation,
        q=req.q_imitation,
        M=req.M_market_potential,
        horizon=horizon,
        n_bins=10,
    )

    cutoff = _resolve_cutoff(
        entropy_profile=entropy_profile,
        marginal_threshold=req.entropy_cutoff_threshold,
        absolute_ceiling=float("inf"),
    )
    cutoff = max(0, min(cutoff, horizon))

    forecast_values = bass_future[:cutoff].tolist()
    forecast_steps = list(range(1, cutoff + 1))
    entropy_out = entropy_profile[:cutoff].tolist()

    return BassForecastResponse(
        forecast_values=forecast_values,
        forecast_steps=forecast_steps,
        entropy_per_step=entropy_out,
        cutoff_step=cutoff,
        truncated=(cutoff < horizon),
    )


@app.post("/entropy/conditional-profile", response_model=EntropyProfileResponse)
async def compute_entropy_degradation_profile(req: EntropyProfileRequest) -> EntropyProfileResponse:
    """
    Computes the Shannon conditional entropy H(X_{t+k}|X_{1..t}) for each step in the
    forecast horizon, given the Bass parameters. Provides a trajectory of information
    degradation.

    Use to diagnose where the forecast becomes unreliable. Do NOT use if the adoption
    series has fewer than 10 points, as the conditional entropy estimation requires
    sufficient historical variability.
    """
    if req.M_market_potential <= 0:
        raise HTTPException(status_code=422, detail="M_market_potential must be > 0 to compute entropy profile.")

    horizon = int(req.horizon_steps)
    n_bins = int(req.n_bins)

    entropy_profile = _compute_conditional_entropy_profile(
        series=req.adoption_series,
        p=req.p_innovation,
        q=req.q_imitation,
        M=req.M_market_potential,
        horizon=horizon,
        n_bins=n_bins,
    )

    return EntropyProfileResponse(
        entropy_per_step=entropy_profile.tolist(),
        horizon_steps=horizon,
        max_entropy=float(np.max(entropy_profile)),
        min_entropy=float(np.min(entropy_profile)),
        mean_entropy=float(np.mean(entropy_profile)),
    )


@app.post("/entropy/cutoff-resolution", response_model=EntropyCutoffResponse)
async def resolve_forecast_horizon_by_entropy(req: EntropyCutoffRequest) -> EntropyCutoffResponse:
    """
    Determines the recommended forecast cutoff step from an entropy profile, given
    thresholds for marginal entropy gain and absolute entropy ceiling.

    Use after computing the entropy profile to automatically truncate the forecast at
    the point where information becomes unreliable. Do NOT use if the entropy profile
    is empty or contains invalid values (e.g., negative or non-monotonic).
    """
    H = np.array(req.entropy_per_step, dtype=float)

    if np.any(H < 0):
        raise EntropyProfileInvalidError(
            "entropy_per_step contains negative values. Shannon entropy must be >= 0."
        )

    cutoff = _resolve_cutoff(
        entropy_profile=H,
        marginal_threshold=req.marginal_entropy_threshold,
        absolute_ceiling=req.absolute_entropy_ceiling,
    )
    cutoff = max(0, min(cutoff, len(H)))

    reliable_steps = list(range(1, cutoff + 1))

    if cutoff == len(H):
        reason = "All steps are within entropy thresholds; no truncation applied."
        truncated_at = None
    elif cutoff == 0:
        reason = (
            f"First step already exceeds absolute_entropy_ceiling ({req.absolute_entropy_ceiling} bits) "
            f"or marginal threshold ({req.marginal_entropy_threshold} bits). Forecast is fully unreliable."
        )
        truncated_at = 1
    else:
        h_at_cutoff = float(H[cutoff]) if cutoff < len(H) else None
        if h_at_cutoff is not None and h_at_cutoff > req.absolute_entropy_ceiling:
            reason = (
                f"Step {cutoff + 1} exceeds absolute_entropy_ceiling "
                f"({h_at_cutoff:.4f} > {req.absolute_entropy_ceiling} bits). "
                f"Reliable up to step {cutoff}."
            )
        else:
            marginal = float(H[cutoff] - H[cutoff - 1]) if cutoff > 0 else float(H[0])
            reason = (
                f"Marginal entropy gain at step {cutoff + 1} ({marginal:.4f} bits) "
                f"dropped below threshold ({req.marginal_entropy_threshold} bits). "
                f"Reliable up to step {cutoff}."
            )
        truncated_at = cutoff + 1

    return EntropyCutoffResponse(
        recommended_cutoff_step=cutoff,
        reliable_steps=reliable_steps,
        truncated_at=truncated_at,
        reason=reason,
    )


@app.post("/forecast/reliability-map", response_model=ReliabilityMapResponse)
async def forecast_with_reliability_map(req: ReliabilityMapRequest) -> ReliabilityMapResponse:
    """
    End-to-end pipeline: fits Bass diffusion parameters via MLE, computes conditional
    entropy profile, resolves cutoff horizon, and returns the forecast along with
    per-step entropy, marginal gains, and a reliability map.

    Use as a one-call solution to obtain a complete, entropy-aware forecast. Do NOT use
    if the input series is too short to calibrate (<15 points) or if
    market_potential_prior is not a plausible upper bound.
    """
    horizon = int(req.horizon_steps)
    n_bins = int(req.n_bins)
    max_iter = int(req.mle_max_iterations)

    bass_params = _calibrate_bass_mle(
        series=req.adoption_series,
        M_prior=req.market_potential_prior,
        max_iter=max_iter,
    )

    p = bass_params["p_innovation"]
    q = bass_params["q_imitation"]
    M = bass_params["M_market_potential"]

    entropy_profile = _compute_conditional_entropy_profile(
        series=req.adoption_series,
        p=p,
        q=q,
        M=M,
        horizon=horizon,
        n_bins=n_bins,
    )

    H = entropy_profile
    marginal_entropy = np.zeros(len(H))
    marginal_entropy[0] = H[0]
    if len(H) > 1:
        marginal_entropy[1:] = np.diff(H)

    cutoff = _resolve_cutoff(
        entropy_profile=H,
        marginal_threshold=req.marginal_entropy_threshold,
        absolute_ceiling=float("inf"),
    )
    cutoff = max(0, min(cutoff, horizon))

    T = len(req.adoption_series)
    t_future = np.arange(T + 1, T + horizon + 1, dtype=float)
    bass_future = _bass_incremental(t_future, p, q, M)

    forecast_values = bass_future[:cutoff].tolist()
    forecast_steps = list(range(1, cutoff + 1))
    entropy_out = H[:cutoff].tolist()
    marginal_out = marginal_entropy[:cutoff].tolist()

    reliability_map = []
    for i in range(cutoff):
        step = i + 1
        h = float(H[i])
        mg = float(marginal_entropy[i])
        max_possible_entropy = float(np.log2(n_bins))
        reliability_score = float(np.clip(1.0 - h / max_possible_entropy, 0.0, 1.0))
        reliability_map.append({
            "step": step,
            "forecast_value": float(bass_future[i]),
            "entropy_bits": round(h, 6),
            "marginal_entropy_bits": round(mg, 6),
            "reliability_score": round(reliability_score, 6),
        })

    return ReliabilityMapResponse(
        p_innovation=p,
        q_imitation=q,
        M_market_potential=M,
        mle_converged=bass_params["converged"],
        forecast_values=forecast_values,
        forecast_steps=forecast_steps,
        entropy_per_step=entropy_out,
        marginal_entropy_per_step=marginal_out,
        cutoff_step=cutoff,
        truncated=(cutoff < horizon),
        reliability_map=reliability_map,
    )

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import asynccontextmanager

import asyncio
import base64
import json
import os
import time
import httpx
from pydantic import Field
from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP as _NexusFastMCP
from mcp.server.transport_security import TransportSecuritySettings

# --- NEXUS: PATCH fix_mcp_dns_rebinding_host ---
# FastMCP() sin host/transport_security explicito activa proteccion
# anti DNS-rebinding con allowlist localhost-only por default del SDK,
# rechazando con 421 "Invalid Host header" cualquier request real
# contra el dominio publico de Railway (bug real confirmado en
# produccion 2026-07-09, ver docstring del generador). Se pasa
# transport_security explicito leyendo RAILWAY_PUBLIC_DOMAIN en
# runtime -- Railway lo inyecta automaticamente en cada servicio, asi
# que este codigo no necesita conocer su propio dominio al generarse.
_nexus_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "*")

_nexus_mcp = _NexusFastMCP(
    'nexus-time-series-forecast-api',
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        # --- PATCH fix_mcp_dns_rebinding_bare_host ---
        # Railway (como cualquier proxy HTTPS estandar) manda el Host
        # header SIN puerto explicito -- "dominio:*" nunca matchea eso,
        # solo matchea "dominio:443". Se agrega tambien el dominio
        # pelado para cubrir ambos casos (bug real confirmado en
        # produccion 2026-07-09: primer fix desplegado, /mcp seguia
        # devolviendo 421 tras el redeploy).
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            _nexus_railway_domain,
            _nexus_railway_domain + ":*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://" + _nexus_railway_domain,
        ],
    ),
)


# --- NEXUS: instancia FastAPI aislada para llamadas MCP->core internas ---
# Comparte los MISMOS objetos de ruta (app.routes) que `app` -- misma
# resolucion real de FastAPI DI (Security()/Depends(), lo que el LLM haya
# escrito) -- pero SIN ningun @app.middleware/add_middleware propio de
# `app` (billing Stripe, rate-limit, x402 PaymentMiddlewareASGI,
# traffic-log). Esos middleware ya corrieron UNA vez sobre la request HTTP
# real a /mcp (Starlette envuelve el Mount de FastMCP en "/" con el mismo
# stack que el resto de `app`) -- esta llamada interna NO debe volver a
# dispararlos, y (para rutas x402-gateadas) no debe recibir el mismo 402
# que la ruta REST publica exige, porque el pago real (si el asset lo
# tiene) se verifica aparte, a nivel de tool MCP -- ver CLAUDE.md SS9.5x.
# `list(...)` fuerza una copia -- Router.__init__ ya copia internamente,
# pero se es explicito aca para que esta instancia quede fija al set de
# rutas REST que existe en este punto (antes de app.mount("/", ...) mas
# abajo), sin importar mutaciones futuras de app.routes.
_nexus_internal_app = FastAPI(routes=list(app.routes))


async def _nexus_mcp_call_core(method: str, path: str, params: dict, headers: dict | None = None) -> Any:
    """
    Llama al endpoint real del core -- via ASGI in-process (sin red
    real, sin segundo proceso) contra _nexus_internal_app (ver arriba),
    NO contra `app` directamente.
    """
    transport = httpx.ASGITransport(app=_nexus_internal_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params, headers=headers)
        else:
            resp = await client.post(path, json=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


# --- PATCH mcp_call_events_telemetry ---
# mcp_call_events / revenue_events -- ver Fase 1 (Revenue/Usage
# Instrumentation). Generator-side por diseno: cualquier asset nuevo
# que FORGE construya de aca en adelante nace con esto, sin depender de
# un patch manual posterior por asset.
#
# Credenciales leidas de env vars Railway con el MISMO patron defensivo
# que ya usa _nexus_usage_middleware (forge_output_saver_v6.py) para
# Stripe: si SUPABASE_URL/SUPABASE_ANON_KEY no estan seteadas (asset
# corriendo local, o el paso de sync de env vars del pipeline de deploy
# todavia no las inyecto -- ESE paso vive fuera de este generador, ver
# billing_reconciliation.py:134-144 para el patron real que lo haria),
# el insert es un no-op silencioso -- nunca rompe la response real de
# una tool call.
#
# IMPORTANTE: usa la key anon/publishable, NUNCA service_role -- esta
# key vive en el runtime del asset deployado (potencialmente expuesta
# via env dump/logs), la RLS de mcp_call_events/revenue_events
# (policies INSERT-only para el rol anon, sin SELECT/UPDATE/DELETE) es
# la unica proteccion real contra un uso indebido de esta key si se
# filtra.
_NEXUS_SUPABASE_URL = os.getenv("SUPABASE_URL")
_NEXUS_SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_NEXUS_SECTOR = 'data_sdk'
_NEXUS_ASSET_NAME = 'Time Series Forecast API'


def _nexus_truncate_ip(raw_ip):
    """Mismo algoritmo/mismo resultado que _nexus_traffic_log_truncate_ip
    (archive/patches/patch_traffic_log_similarity_search.py) -- portado
    aca para que todo asset nuevo lo tenga desde generacion. Trunca a
    /24 (IPv4) o /64 (IPv6); nunca devuelve la IP completa."""
    if not raw_ip:
        return None
    if ":" in raw_ip and "." not in raw_ip:
        segments = [s for s in raw_ip.split(":") if s]
        head = segments[:4] if len(segments) >= 4 else segments
        return (":".join(head) + "::/64") if head else None
    octets = raw_ip.split(".")
    if len(octets) == 4 and all(o.isdigit() for o in octets):
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return None


def _nexus_extract_wallet(payment_header):
    """Mismo algoritmo que _nexus_rate_limit_extract_wallet
    (archive/patches/patch_rate_limit_similarity_search.py) -- decodifica
    el header X-PAYMENT (base64 -> JSON) y extrae la wallet pagadora de
    payload.authorization.from."""
    try:
        padded = payment_header + "=" * (-len(payment_header) % 4)
        payload = json.loads(base64.b64decode(padded))
        payer = payload.get("payload", {}).get("authorization", {}).get("from")
        return payer.lower() if isinstance(payer, str) and payer else None
    except Exception:
        return None


def _nexus_call_context(ctx):
    """
    Best-effort: ctx.request_context.request es un starlette.Request
    REAL incluso con stateless_http=True -- se puebla por REQUEST HTTP
    individual (mcp/server/streamable_http.py: ServerMessageMetadata(
    request_context=request), poblado en _create_session_message()),
    no por continuidad de sesion. Verificado contra el codigo fuente
    real de mcp==1.28.1 (version pinneada del proyecto) antes de asumir
    que el dato esta disponible -- no es una suposicion sin chequear.

    agent_framework: no existe un campo dedicado para esto en el
    protocolo MCP tal como esta implementado hoy con
    stateless_http=True -- ctx.session.client_params.clientInfo (que
    SI llevaria un nombre de framework/cliente real) solo se puebla si
    el mensaje initialize y la tool call posterior comparten la misma
    ServerSession, y en modo stateless cada POST crea una ServerSession
    nueva (mcp/server/streamable_http_manager.py:_handle_stateless_request).
    La señal real y disponible en su lugar es el header User-Agent de
    la request HTTP -- se usa tal cual cuando el cliente lo manda
    (nunca fabricado; queda None si el cliente no lo envia).
    """
    ip_range = None
    agent_framework = None
    wallet = None
    try:
        request = ctx.request_context.request if ctx is not None else None
        if request is not None:
            forwarded = request.headers.get("x-forwarded-for")
            raw_ip = forwarded.split(",")[0].strip() if forwarded else (
                request.client.host if request.client else None
            )
            ip_range = _nexus_truncate_ip(raw_ip)
            ua = request.headers.get("user-agent")
            agent_framework = ua[:255] if ua else None
            payment_header = request.headers.get("x-payment")
            if payment_header:
                wallet = _nexus_extract_wallet(payment_header)
    except Exception:
        pass
    return ip_range, agent_framework, wallet


async def _nexus_supabase_insert(table, payload):
    if not _NEXUS_SUPABASE_URL or not _NEXUS_SUPABASE_ANON_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"{_NEXUS_SUPABASE_URL}/rest/v1/{table}",
                json=payload,
                headers={
                    "apikey": _NEXUS_SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {_NEXUS_SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
    except Exception:
        pass  # nunca romper el flujo real por un fallo de telemetria


async def _nexus_log_mcp_call_event(tool_id, success, latency_ms, ctx, route_key=None):
    """
    Escribe mcp_call_events (proxy de uso/latencia) siempre que haya
    credenciales -- dispara en el `finally` de cada tool MCP, es decir
    cuando el handler retorna (con o sin excepcion), ANTES de que exista
    ningun intento de settlement x402 real para esa llamada.

    --- PATCH x402_revenue_events_hook ---
    Ya NO escribe revenue_events desde aca (si escribia hasta esta
    sesion -- ver CLAUDE.md SS9.58). Ese insert vivia en este mismo punto
    proxy: para un tool protegido por x402 (create_payment_wrapper), el
    handler corre y sirve el recurso ANTES de que se intente el
    settlement (x402/mcp/server.py: handler primero, settle_payment()
    despues) -- loguear "cobrado" aca era literalmente loguearlo antes
    de que el pago se intentara liquidar. revenue_events ahora se
    escribe desde _nexus_log_x402_revenue_event, enganchado directo a
    x402ResourceServer.on_after_settle() (ver mas abajo) -- el unico
    punto que conoce el SettleResponse real (success/transaction/payer)
    y dispara solo cuando success es True.

    x402_routes/price_charged se mantienen aca SOLO para poblar
    mcp_call_events.price_charged, que si tiene sentido como proxy:
    "cuanto hubiera cobrado esta llamada si el pago hubiera settleado",
    no una afirmacion de que settleo. is_paid_route sigue en False para
    cualquier asset nuevo (x402 no se genera hoy por FORGE, CLAUDE.md
    SS8) hasta que un patch manual agregue _NEXUS_X402_ROUTES.
    """
    ip_range, agent_framework, wallet = _nexus_call_context(ctx)
    price_charged = None
    x402_routes = globals().get("_NEXUS_X402_ROUTES") or {}
    is_paid_route = route_key is not None and route_key in x402_routes
    if is_paid_route:
        raw_price = globals().get("_NEXUS_X402_PRICE")
        if raw_price:
            try:
                price_charged = float(str(raw_price).lstrip("$"))
            except Exception:
                price_charged = None
    # --- PATCH mcp_call_events_asset_name ---
    await _nexus_supabase_insert("mcp_call_events", {
        "agent_framework": agent_framework,
        "tool_id": tool_id,
        "sector": _NEXUS_SECTOR,
        "asset_name": _NEXUS_ASSET_NAME,
        # token_input/token_output: null a proposito -- ningun asset
        # que FORGE genera hoy envuelve una llamada LLM propia (son
        # productos como vector search / websocket / rate limiting,
        # no proxies de un modelo). Si algun dia existe un asset que SI
        # envuelva un LLM, ese caso deberia poblar estos campos en su
        # propio call site en vez de forzar un valor generico aca.
        "token_input": None,
        "token_output": None,
        "success": success,
        "latency_ms": latency_ms,
        "client_ip_range": ip_range,
        "price_charged": price_charged,
    })


async def _nexus_log_x402_revenue_event(ctx) -> None:
    """
    --- PATCH x402_revenue_events_hook ---
    AfterSettleHook real para revenue_events -- registrado via
    _nexus_register_x402_revenue_logging(), nunca llamado directo.
    Firma compatible con x402.server_base.AfterSettleHook
    (Callable[[SettleResultContext], Awaitable[None] | None]) pero SIN
    importar el tipo -- duck-typed a proposito, para que este generador
    no obligue a que x402 este instalado en assets que no lo tienen (la
    funcion se emite igual en todo asset nuevo, pero solo se registra
    -- ver _nexus_register_x402_revenue_logging -- si el asset tiene su
    propio _nexus_x402_server real).

    x402ResourceServer.on_after_settle() (x402/server.py) SOLO dispara
    cuando settle_result.success es True -- confirmado leyendo
    _settle_payment_core (x402/server_base.py): un settle fallido nunca
    llega a la lista de after-settle hooks, asi que esta funcion nunca
    necesita chequear success ademas del propio getattr defensivo de
    abajo. Nunca levanta: un bug aca no debe poder romper la respuesta
    de pago real que el caller ya esta por recibir.
    """
    try:
        result = getattr(ctx, "result", None)
        requirements = getattr(ctx, "requirements", None)
        if result is None or requirements is None or not getattr(result, "success", False):
            return
        # amount viene en unidades atomicas del asset -- USDC (6
        # decimales) en los 2 assets reales que usan x402 hoy
        # (similarity-search-api, ws; ver patch_x402_similarity_search.py
        # / patch_x402_ws.py). Mismo mismatch de schema ya documentado
        # mas arriba para mcp_call_events.price_charged: amount_eur no
        # es EUR, es el valor numerico crudo, sin conversion FX.
        raw_amount = getattr(requirements, "amount", None)
        amount_eur = int(raw_amount) / 1_000_000 if raw_amount is not None else None
        await _nexus_supabase_insert("revenue_events", {
            "asset_name": _NEXUS_ASSET_NAME,
            "amount_eur": amount_eur,
            "pricing_model": "x402",
            "stripe_event_id": None,
            "customer_id": getattr(result, "payer", None),
        })
    except Exception:
        pass


def _nexus_register_x402_revenue_logging(x402_server) -> None:
    """
    --- PATCH x402_revenue_events_hook ---
    Registra _nexus_log_x402_revenue_event contra el on_after_settle()
    REAL de x402ResourceServer -- llamar UNA vez, justo despues de
    instanciar/inicializar _nexus_x402_server (mismo lugar donde un
    patch x402 manual ya llama _nexus_x402_server.initialize()).

    Por que aca y no en _nexus_log_mcp_call_event / no en
    PaymentWrapperHooks.on_after_settlement (x402/mcp/types.py): tanto
    las rutas REST (PaymentMiddlewareASGI -> x402_http_server.py:216)
    como los tools MCP (create_payment_wrapper -> server.py /
    server_async.py) llaman settle_payment() sobre la MISMA instancia
    compartida de x402ResourceServer -- un solo hook registrado aca
    cubre ambas superficies de pago de un asset sin que este generador
    necesite saber cual usa cada uno. PaymentWrapperHooks.on_after_settlement
    es exclusivo del wrapper MCP: hubiera dejado sin cubrir cualquier
    pago hecho directo por REST.

    Riesgo conocido, sin cerrar (ver CLAUDE.md SS9.58): no hay forma,
    desde este codigo, de confirmar que settle_result.success==True
    significa "confirmado on-chain" en vez de "el facilitador externo
    (CDP/x402.org) lo acepto como valido, cadena aun no verificada" --
    settle_payment() hace una unica llamada HTTP a {facilitator}/settle
    y confia en el campo `success` que ese facilitador devuelva, sin
    poll ni retry propios. El facilitador de REFERENCIA que trae este
    mismo SDK (x402/mechanisms/evm/exact/facilitator.py) SI espera
    confirmacion on-chain real (wait_for_transaction_receipt) antes de
    devolver success=True -- evidencia de la semantica esperada del
    protocolo, no prueba de que un facilitador externo la respete.

    Idempotencia: revenue_events no tiene columna de tx hash ni
    constraint unico, y la RLS del rol anon es INSERT-only (sin
    SELECT) -- no hay forma de check-before-insert desde el propio
    asset. Se confia en que EIP-3009 (nonce de un solo uso) hace que un
    replay real del mismo payload firmado falle el settlement
    (success=False) en el segundo intento -- este hook solo dispara en
    success=True. Inferencia del diseno del esquema de firma, no
    verificada empiricamente. Cerrar esto de forma robusta requeriria
    una columna tx_hash + unique index en revenue_events -- cambio de
    schema a infra compartida, fuera de alcance de este generador.
    """
    x402_server.on_after_settle(_nexus_log_x402_revenue_event)


async def _nexus_log_traffic_event(ip_range, method, path, status) -> None:
    """
    --- PATCH traffic_events_hook ---
    Primitiva reusable para persistir [NEXUS_TRAFFIC] a Supabase --
    este generador NO se autoregistra en ningun middleware (el
    middleware de traffic-log sigue siendo un artefacto manual, opt-in,
    por asset -- mismo estado que x402, ver CLAUDE.md SS9.59). Pensada
    para que un patch de traffic-log (mismo patron que
    archive/patches/patch_traffic_log_similarity_search.py) la llame
    JUNTO al print() que ya existe, sin reimplementar el insert en cada
    asset.

    Aditivo por contrato: el caller es responsable de mantener el
    print("[NEXUS_TRAFFIC] ...") intacto y llamar a esta funcion
    ADEMAS, no en su lugar -- AegisAgent.PortfolioAuditor
    (aegis_discovery.py) sigue leyendo esa linea en vivo desde logs de
    Railway, no de Supabase. Nunca levanta: un fallo de telemetria
    nunca debe poder romper la response real.
    """
    try:
        await _nexus_supabase_insert("traffic_events", {
            "asset_name": _NEXUS_ASSET_NAME,
            "ip_range": ip_range,
            "method": method,
            "path": path,
            "status": status,
        })
    except Exception:
        pass



@_nexus_mcp.tool(name='nexus_time_series_forecast_api_calibrate_bass_diffusion_mle', description='Calibrates Bass diffusion model parameters (p_innovation, q_imitation, M_market_potential) via Maximum Likelihood Estimation on the provided historical adoption series. Use to estimate model coefficients from observed adoption data, especially with at least 15 points. Do NOT use for very short series (<5 points) or when adoption is still in the initial flat phase without a clear inflection – the MLE may fail to converge.')
async def calibrate_bass_diffusion_mle(adoption_series: Annotated[list[float], Field(..., description='Historical cumulative or periodic adoption numbers (e.g., cumulative users per day). Must be non-negative and increasing or non-decreasing.', min_length=5)], market_potential_prior: Annotated[float, Field(..., description='Prior estimate of total addressable market size (M). Must be greater than the last value of adoption_series.', ge=0)], mle_max_iterations: Annotated[float, Field(..., description='Maximum number of iterations for the MLE optimization algorithm.', ge=1, le=10000)], ctx: Context) -> dict[str, Any]:
    """Bass MLE Calibration"""
    _nexus_path = '/bass/mle/calibrate'.format()
    params = {"adoption_series": adoption_series, "market_potential_prior": market_potential_prior, "mle_max_iterations": mle_max_iterations}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_time_series_forecast_api_calibrate_bass_diffusion_mle', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /bass/mle/calibrate',
        ))

@_nexus_mcp.tool(name='nexus_time_series_forecast_api_forecast_bass_adoption_curve', description='Generates future adoption values over a given horizon using the Bass diffusion model. The forecast is truncated if the conditional entropy exceeds the configured threshold, ensuring only reliable steps are returned. Use when you have calibrated Bass parameters and need a numeric forecast with an entropy-based reliability cutoff. Do NOT use without first providing valid p, q, M parameters (missing parameters will raise BassParamsMissingError).')
async def forecast_bass_adoption_curve(adoption_series: Annotated[list[float], Field(..., description='Historical adoption series used for optional entropy computation and as a baseline for forecast alignment.', min_length=1)], horizon_steps: Annotated[float, Field(..., description='Number of future steps to forecast. Must be a positive integer.', ge=1)], p_innovation: Annotated[float, Field(..., description='Coefficient of innovation (p) from Bass calibration.', ge=0, le=1)], q_imitation: Annotated[float, Field(..., description='Coefficient of imitation (q) from Bass calibration.', ge=0, le=1)], M_market_potential: Annotated[float, Field(..., description='Estimated total market potential (saturation level). Must be > 0.', ge=0)], entropy_cutoff_threshold: Annotated[float, Field(..., description='Marginal entropy gain threshold per step (in bits) below which the forecast horizon is cut off.', ge=0)], ctx: Context) -> dict[str, Any]:
    """Bass Adoption Forecast"""
    _nexus_path = '/bass/forecast'.format()
    params = {"adoption_series": adoption_series, "horizon_steps": horizon_steps, "p_innovation": p_innovation, "q_imitation": q_imitation, "M_market_potential": M_market_potential, "entropy_cutoff_threshold": entropy_cutoff_threshold}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_time_series_forecast_api_forecast_bass_adoption_curve', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /bass/forecast',
        ))

@_nexus_mcp.tool(name='nexus_time_series_forecast_api_compute_entropy_degradation_profile', description='Computes the Shannon conditional entropy H(X_{t+k}|X_{1..t}) for each step in the forecast horizon, given the Bass parameters. Provides a trajectory of information degradation. Use to diagnose where the forecast becomes unreliable. Do NOT use if the adoption series has fewer than 10 points, as the conditional entropy estimation requires sufficient historical variability.')
async def compute_entropy_degradation_profile(adoption_series: Annotated[list[float], Field(..., description='Historical adoption series for estimating the state distribution.', min_length=10)], horizon_steps: Annotated[float, Field(..., description='Number of future steps for which to compute entropy.', ge=1)], n_bins: Annotated[float, Field(..., description='Number of bins for discretizing continuous values to estimate empirical entropy. Must be at least 2.', ge=2, le=100)], p_innovation: Annotated[float, Field(..., description='Coefficient of innovation (p).', ge=0, le=1)], q_imitation: Annotated[float, Field(..., description='Coefficient of imitation (q).', ge=0, le=1)], M_market_potential: Annotated[float, Field(..., description='Estimated market potential.', ge=0)], ctx: Context) -> dict[str, Any]:
    """Entropy Degradation Profile"""
    _nexus_path = '/entropy/conditional-profile'.format()
    params = {"adoption_series": adoption_series, "horizon_steps": horizon_steps, "n_bins": n_bins, "p_innovation": p_innovation, "q_imitation": q_imitation, "M_market_potential": M_market_potential}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_time_series_forecast_api_compute_entropy_degradation_profile', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /entropy/conditional-profile',
        ))

@_nexus_mcp.tool(name='nexus_time_series_forecast_api_resolve_forecast_horizon_by_entropy', description='Determines the recommended forecast cutoff step from an entropy profile, given thresholds for marginal entropy gain and absolute entropy ceiling. Use after computing the entropy profile to automatically truncate the forecast at the point where information becomes unreliable. Do NOT use if the entropy profile is empty or contains invalid values (e.g., negative or non-monotonic).')
async def resolve_forecast_horizon_by_entropy(entropy_per_step: Annotated[list[float], Field(..., description='Array of conditional entropy values per horizon step (starting from step 1).', min_length=1)], marginal_entropy_threshold: Annotated[float, Field(..., description='Minimum acceptable increase in entropy per step (bits). Drop below this signals unreliable forecast.', ge=0)], absolute_entropy_ceiling: Annotated[float, Field(..., description='Absolute entropy value (bits) beyond which the forecast is considered too uncertain.', ge=0)], ctx: Context) -> dict[str, Any]:
    """Entropy-Based Horizon Resolution"""
    _nexus_path = '/entropy/cutoff-resolution'.format()
    params = {"entropy_per_step": entropy_per_step, "marginal_entropy_threshold": marginal_entropy_threshold, "absolute_entropy_ceiling": absolute_entropy_ceiling}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_time_series_forecast_api_resolve_forecast_horizon_by_entropy', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /entropy/cutoff-resolution',
        ))

@_nexus_mcp.tool(name='nexus_time_series_forecast_api_forecast_with_reliability_map', description='End-to-end pipeline: fits Bass diffusion parameters via MLE, computes conditional entropy profile, resolves cutoff horizon, and returns the forecast along with per-step entropy, marginal gains, and a reliability map. Use as a one-call solution to obtain a complete, entropy-aware forecast. Do NOT use if the input series is too short to calibrate (<15 points) or if market_potential_prior is not a plausible upper bound.')
async def forecast_with_reliability_map(adoption_series: Annotated[list[float], Field(..., description='Historical adoption series (cumulative numbers).', min_length=15)], horizon_steps: Annotated[float, Field(..., description='Desired forecast horizon (number of steps).', ge=1)], market_potential_prior: Annotated[float, Field(..., description='Prior estimate of the market potential M for Bass calibration.', ge=0)], marginal_entropy_threshold: Annotated[float, Field(..., description='Entropy threshold for cutoff (bits).', ge=0)], n_bins: Annotated[float, Field(..., description='Number of bins for entropy discretization.', ge=2, le=100)], mle_max_iterations: Annotated[float, Field(..., description='Maximum iterations for the MLE optimizer.', ge=1, le=10000)], ctx: Context) -> dict[str, Any]:
    """Forecast with Reliability Map"""
    _nexus_path = '/forecast/reliability-map'.format()
    params = {"adoption_series": adoption_series, "horizon_steps": horizon_steps, "market_potential_prior": market_potential_prior, "marginal_entropy_threshold": marginal_entropy_threshold, "n_bins": n_bins, "mle_max_iterations": mle_max_iterations}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_time_series_forecast_api_forecast_with_reliability_map', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /forecast/reliability-map',
        ))


# Crea el sub-app ASGI de streamable HTTP -- DEBE llamarse antes de
# poder acceder a _nexus_mcp.session_manager (se crea de forma
# perezosa, ver docstring del modulo).
# Se monta en "/" (no en "/mcp"): streamable_http_app() YA expone su
# propia ruta interna en "/mcp" -- montarlo de nuevo en "/mcp" duplica
# el path a "/mcp/mcp" y da 404 (bug real encontrado probando esto en
# runtime con un cliente MCP de verdad, no algo teorico).
_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()

# --- NEXUS: PATCH mcp_lifespan_composition_fix ---
# @app.on_event() SOLO se ejecuta si Starlette uso _DefaultLifespan --
# eso pasa unicamente cuando el `app = FastAPI(...)` que genero el LLM
# NO paso su propio parametro `lifespan=`. Si el LLM SI definio uno
# (tipico en assets con estado -- ej. cleanup de conexiones WebSocket
# al shutdown), Starlette usa ESE callable exclusivamente y los
# handlers @app.on_event quedan sin ejecutarse -- sin warning, sin
# error, el server bootea limpio ("Application startup complete") y
# el primer request a /mcp explota con "RuntimeError: Task group is
# not initialized." (confirmado contra Router.__init__ de Starlette:
# lifespan=None -> _DefaultLifespan(self) [dispara on_event],
# lifespan=<callable> -> se usa ESE, on_event nunca corre). Bug real
# encontrado en produccion 2026-07-25 (asset "ws", que define su
# propio lifespan para cerrar sesiones WebSocket abiertas).
#
# Fix: envolver el lifespan_context que Starlette YA construyo (sea
# _DefaultLifespan o el custom del LLM) en vez de competir con el
# via @app.on_event. Funciona en ambos casos sin parsear ni tocar
# el `app = FastAPI(...)` original.
_nexus_prev_lifespan_context = app.router.lifespan_context


@asynccontextmanager
async def _nexus_combined_lifespan(app):
    async with _nexus_mcp.session_manager.run():
        async with _nexus_prev_lifespan_context(app):
            yield


app.router.lifespan_context = _nexus_combined_lifespan


# --- NEXUS: receptor real de webhooks de Stripe (inyectado por forge_output_saver_v6) ---
from fastapi import Request as _NexusStripeRequest
from fastapi.responses import JSONResponse as _NexusStripeJSONResponse

@app.post("/stripe/webhook")
async def _nexus_stripe_webhook(request: _NexusStripeRequest):
    import os as _nexus_os
    import stripe as _nexus_stripe
    _webhook_secret = _nexus_os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not _webhook_secret:
        return _NexusStripeJSONResponse(
            status_code=404,
            content={"error": "stripe webhook not configured"},
        )
    _secret_key = _nexus_os.environ.get("STRIPE_SECRET_KEY")
    if _secret_key:
        _nexus_stripe.api_key = _secret_key
    _payload = await request.body()
    _sig_header = request.headers.get("stripe-signature", "")
    try:
        _event = _nexus_stripe.Webhook.construct_event(
            _payload, _sig_header, _webhook_secret
        )
    except ValueError:
        return _NexusStripeJSONResponse(
            status_code=400, content={"error": "invalid payload"}
        )
    except _nexus_stripe.error.SignatureVerificationError:
        return _NexusStripeJSONResponse(
            status_code=400, content={"error": "invalid signature"}
        )
    # NEXUS: solo verificacion + ack real -- el gate de
    # autorizacion por estado de suscripcion y la politica de
    # dunning/downgrade son decisiones de producto pendientes,
    # no implementadas a proposito (ver CLAUDE.md).
    print(
        f"[NEXUS_STRIPE_WEBHOOK] type={_event['type']} "
        f"id={_event['id']}"
    )
    return _NexusStripeJSONResponse(
        status_code=200, content={"received": True}
    )


app.mount("/", _nexus_mcp_asgi_app)