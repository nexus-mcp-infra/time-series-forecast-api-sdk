"""
Cliente HTTP para TimeSeriesForecastApi -- generado deterministicamente desde
el contrato OpenAPI real (src/agents/openapi_sdk_generator.py). No
edites rutas/params a mano aca -- se regenera en cada build desde
tool_spec; sdk.js sale del mismo spec, por diseno no puede divergir.
"""
from __future__ import annotations

import requests
from typing import Any, Optional


class TimeSeriesForecastApiError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TimeSeriesForecastApi:
    """HTTP client. Base URL real del deploy: https://time-series-forecast-api.railway.app"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://time-series-forecast-api.railway.app', timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})

    def calibrate_bass_diffusion_mle(self, adoption_series: list[float], market_potential_prior: float, mle_max_iterations: float) -> dict:
        """Calibrates Bass diffusion model parameters (p_innovation, q_imitation, M_market_potential) via Maximum Likelihood Estimation on the provided historical adoption series. Use to estimate model coefficients from observed adoption data, especially with at least 15 points. Do NOT use for very short series (<5 points) or when adoption is still in the initial flat phase without a clear inflection – the MLE may fail to converge.

        Calls POST /bass/mle/calibrate
        """
        payload = {}
        payload['adoption_series'] = adoption_series
        payload['market_potential_prior'] = market_potential_prior
        payload['mle_max_iterations'] = mle_max_iterations
        url = self.base_url + '/bass/mle/calibrate'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise TimeSeriesForecastApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def forecast_bass_adoption_curve(self, adoption_series: list[float], horizon_steps: float, p_innovation: float, q_imitation: float, M_market_potential: float, entropy_cutoff_threshold: float) -> dict:
        """Generates future adoption values over a given horizon using the Bass diffusion model. The forecast is truncated if the conditional entropy exceeds the configured threshold, ensuring only reliable steps are returned. Use when you have calibrated Bass parameters and need a numeric forecast with an entropy-based reliability cutoff. Do NOT use without first providing valid p, q, M parameters (missing parameters will raise BassParamsMissingError).

        Calls POST /bass/forecast
        """
        payload = {}
        payload['adoption_series'] = adoption_series
        payload['horizon_steps'] = horizon_steps
        payload['p_innovation'] = p_innovation
        payload['q_imitation'] = q_imitation
        payload['M_market_potential'] = M_market_potential
        payload['entropy_cutoff_threshold'] = entropy_cutoff_threshold
        url = self.base_url + '/bass/forecast'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise TimeSeriesForecastApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def compute_entropy_degradation_profile(self, adoption_series: list[float], horizon_steps: float, n_bins: float, p_innovation: float, q_imitation: float, M_market_potential: float) -> dict:
        """Computes the Shannon conditional entropy H(X_{t+k}|X_{1..t}) for each step in the forecast horizon, given the Bass parameters. Provides a trajectory of information degradation. Use to diagnose where the forecast becomes unreliable. Do NOT use if the adoption series has fewer than 10 points, as the conditional entropy estimation requires sufficient historical variability.

        Calls POST /entropy/conditional-profile
        """
        payload = {}
        payload['adoption_series'] = adoption_series
        payload['horizon_steps'] = horizon_steps
        payload['n_bins'] = n_bins
        payload['p_innovation'] = p_innovation
        payload['q_imitation'] = q_imitation
        payload['M_market_potential'] = M_market_potential
        url = self.base_url + '/entropy/conditional-profile'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise TimeSeriesForecastApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def resolve_forecast_horizon_by_entropy(self, entropy_per_step: list[float], marginal_entropy_threshold: float, absolute_entropy_ceiling: float) -> dict:
        """Determines the recommended forecast cutoff step from an entropy profile, given thresholds for marginal entropy gain and absolute entropy ceiling. Use after computing the entropy profile to automatically truncate the forecast at the point where information becomes unreliable. Do NOT use if the entropy profile is empty or contains invalid values (e.g., negative or non-monotonic).

        Calls POST /entropy/cutoff-resolution
        """
        payload = {}
        payload['entropy_per_step'] = entropy_per_step
        payload['marginal_entropy_threshold'] = marginal_entropy_threshold
        payload['absolute_entropy_ceiling'] = absolute_entropy_ceiling
        url = self.base_url + '/entropy/cutoff-resolution'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise TimeSeriesForecastApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def forecast_with_reliability_map(self, adoption_series: list[float], horizon_steps: float, market_potential_prior: float, marginal_entropy_threshold: float, n_bins: float, mle_max_iterations: float) -> dict:
        """End-to-end pipeline: fits Bass diffusion parameters via MLE, computes conditional entropy profile, resolves cutoff horizon, and returns the forecast along with per-step entropy, marginal gains, and a reliability map. Use as a one-call solution to obtain a complete, entropy-aware forecast. Do NOT use if the input series is too short to calibrate (<15 points) or if market_potential_prior is not a plausible upper bound.

        Calls POST /forecast/reliability-map
        """
        payload = {}
        payload['adoption_series'] = adoption_series
        payload['horizon_steps'] = horizon_steps
        payload['market_potential_prior'] = market_potential_prior
        payload['marginal_entropy_threshold'] = marginal_entropy_threshold
        payload['n_bins'] = n_bins
        payload['mle_max_iterations'] = mle_max_iterations
        url = self.base_url + '/forecast/reliability-map'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise TimeSeriesForecastApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()