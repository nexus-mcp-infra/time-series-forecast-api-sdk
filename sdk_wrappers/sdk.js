/**
 * Cliente HTTP para TimeSeriesForecastApi -- generado deterministicamente
 * desde el contrato OpenAPI real (src/agents/openapi_sdk_generator.py).
 * No edites rutas/params a mano aca -- sdk.py sale del mismo spec,
 * por diseno no puede divergir.
 */

class TimeSeriesForecastApiError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'TimeSeriesForecastApiError';
    this.statusCode = statusCode;
  }
}

class TimeSeriesForecastApi {
  constructor(apiKey, baseUrl = "https://time-series-forecast-api.railway.app", timeoutMs = 30000) {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
  }

  _headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.apiKey) h['X-API-Key'] = this.apiKey;
    return h;
  }

  async calibrateBassDiffusionMle({ adoption_series, market_potential_prior, mle_max_iterations }) {
    // Calibrates Bass diffusion model parameters (p_innovation, q_imitation, M_market_potential) via Maximum Likelihood Estimation on the provided historical adoption series. Use to estimate model coefficie
    // Calls POST /bass/mle/calibrate
    const payload = { adoption_series, market_potential_prior, mle_max_iterations };
    const url = `${this.baseUrl}/bass/mle/calibrate`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new TimeSeriesForecastApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async forecastBassAdoptionCurve({ adoption_series, horizon_steps, p_innovation, q_imitation, M_market_potential, entropy_cutoff_threshold }) {
    // Generates future adoption values over a given horizon using the Bass diffusion model. The forecast is truncated if the conditional entropy exceeds the configured threshold, ensuring only reliable step
    // Calls POST /bass/forecast
    const payload = { adoption_series, horizon_steps, p_innovation, q_imitation, M_market_potential, entropy_cutoff_threshold };
    const url = `${this.baseUrl}/bass/forecast`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new TimeSeriesForecastApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async computeEntropyDegradationProfile({ adoption_series, horizon_steps, n_bins, p_innovation, q_imitation, M_market_potential }) {
    // Computes the Shannon conditional entropy H(X_{t+k}|X_{1..t}) for each step in the forecast horizon, given the Bass parameters. Provides a trajectory of information degradation. Use to diagnose where t
    // Calls POST /entropy/conditional-profile
    const payload = { adoption_series, horizon_steps, n_bins, p_innovation, q_imitation, M_market_potential };
    const url = `${this.baseUrl}/entropy/conditional-profile`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new TimeSeriesForecastApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async resolveForecastHorizonByEntropy({ entropy_per_step, marginal_entropy_threshold, absolute_entropy_ceiling }) {
    // Determines the recommended forecast cutoff step from an entropy profile, given thresholds for marginal entropy gain and absolute entropy ceiling. Use after computing the entropy profile to automatical
    // Calls POST /entropy/cutoff-resolution
    const payload = { entropy_per_step, marginal_entropy_threshold, absolute_entropy_ceiling };
    const url = `${this.baseUrl}/entropy/cutoff-resolution`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new TimeSeriesForecastApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async forecastWithReliabilityMap({ adoption_series, horizon_steps, market_potential_prior, marginal_entropy_threshold, n_bins, mle_max_iterations }) {
    // End-to-end pipeline: fits Bass diffusion parameters via MLE, computes conditional entropy profile, resolves cutoff horizon, and returns the forecast along with per-step entropy, marginal gains, and a 
    // Calls POST /forecast/reliability-map
    const payload = { adoption_series, horizon_steps, market_potential_prior, marginal_entropy_threshold, n_bins, mle_max_iterations };
    const url = `${this.baseUrl}/forecast/reliability-map`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new TimeSeriesForecastApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

}

module.exports = { TimeSeriesForecastApi, TimeSeriesForecastApiError };