# README

## FastAPI Service for Bass Model Forecasting and Entropy Analysis

This service provides endpoints for calibrating the Bass diffusion model, forecasting adoption rates, computing entropy profiles, determining optimal cutoff steps based on entropy thresholds, and generating reliability maps. The service is built using FastAPI, Python 3.11+, ASGI, and Uvicorn.

### Key Features
1. **Bass Model Calibration**: Calibrates the Bass model using Maximum Likelihood Estimation (MLE) to fit a given adoption series.
2. **Forecasting**: Provides both numerical forecasts of adoption rates and an entropy map that helps in determining the reliability of predictions over time.
3. **Entropy Analysis**: Computes entropy profiles and recommends cutoff steps based on user-defined thresholds.

### Example Usage

#### 1. Calibrating the Bass Model
**Endpoint**: `/bass/calibrate`

**Method**: `POST`

**Request Body**:
```json
{
    "adoption_series": [10, 20, 30, 40, 50],
    "market_potential_prior": 100,
    "mle_max_iterations": 1000
}
```

**Response Body**:
```json
{
    "p_innovation": 0.02,
    "q_imitation": 0.35,
    "M_market_potential": 100,
    "nll": 2.4,
    "converged": true
}
```

#### 2. Forecasting with Entropy Analysis
**Endpoint**: `/bass/forecast`

**Method**: `POST`

**Request Body**:
```json
{
    "adoption_series": [10, 20, 30, 40, 50],
    "horizon_steps": 10,
    "p_innovation": 0.02,
    "q_imitation": 0.35,
    "M_market_potential": 100,
    "entropy_cutoff_threshold": 3.0
}
```

**Response Body**:
```json
{
    "forecast_values": [50, 60, 70, 80, 90, 95, 98, 99, 100, 100],
    "forecast_steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "entropy_per_step": [0.5, 0.7, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
    "cutoff_step": 7,
    "truncated": true
}
```

#### 3. Entropy Profile Calculation
**Endpoint**: `/entropy/profile`

**Method**: `POST`

**Request Body**:
```json
{
    "adoption_series": [10, 20, 30, 40, 50],
    "horizon_steps": 10,
    "n_bins": 20,
    "p_innovation": 0.02,
    "q_imitation": 0.35,
    "M_market_potential": 100
}
```

**Response Body**:
```json
{
    "entropy_per_step": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8],
    "horizon_steps": 10,
    "max_entropy": 1.8,
    "min_entropy": 0.4,
    "mean_entropy": 1.3
}
```

#### 4. Entropy Cutoff Determination
**Endpoint**: `/entropy/cutoff`

**Method**: `POST`

**Request Body**:
```json
{
    "entropy_per_step": [0.5, 0.7, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
    "marginal_entropy_threshold": 0.5,
    "absolute_entropy_ceiling": 3.0
}
```

**Response Body**:
```json
{
    "recommended_cutoff_step": 7,
    "reliable_steps": [1, 2, 3, 4, 5, 6, 7],
    "truncated_at": 7,
    "reason": "Entropy threshold exceeded"
}
```

#### 5. Reliability Map Generation
**Endpoint**: `/reliability/map`

**Method**: `POST`

**Request Body**:
```json
{
    "adoption_series": [10, 20, 30, 40, 50],
    "horizon_steps": 10,
    "market_potential_prior": 100,
    "marginal_entropy_threshold": 1.0,
    "n_bins": 20,
    "mle_max_iterations": 1000
}
```

**Response Body**:
```json
{
    "p_innovation": 0.02,
    "q_imitation": 0.35,
    "M_market_potential": 100,
    "mle_converged": true,
    "forecast_values": [50, 60, 70, 80, 90, 95, 98, 99, 100, 100],
    "forecast_steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "entropy_per_step": [0.5, 0.7, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
    "marginal_entropy_per_step": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "cutoff_step": 9,
    "truncated": false,
    "reliability_map": [
        {"step": 1, "entropy": 0.5, "reliable": true},
        {"step": 2, "entropy": 0.7, "reliable": true},
        {"step": 3, "entropy": 1.0, "reliable": false},
        // ...
        {"step": 10, "entropy": 1.9, "reliable": false}
    ]
}
```

### Models

The service uses Pydantic models to validate and structure the requests and responses. The exact field names must be used as specified in the models.

### Dependencies
- FastAPI: `pip install fastapi`
- Uvicorn: `pip install uvicorn`
- Python 3.11+

### Running the Service

To run the service, use the following command:
```bash
uvicorn main:app --reload
```

Replace `main` with the name of your FastAPI app file.

This README provides a comprehensive overview of the endpoints available in the service, along with example requests and responses. The models ensure that all data is validated and structured correctly, providing robustness to the API interactions.

---

## Pricing

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |

No base fee. No storage fee. No minimum commitment. You pay for computation, not for parking vectors you queried once.