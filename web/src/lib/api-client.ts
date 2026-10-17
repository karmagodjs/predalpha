/**
 * API client for PredAlpha Research Website.
 * Connects to the local FastAPI backend (http://127.0.0.1:8000) or internal Next.js routes.
 */

export interface PredictionResult {
  probabilities: {
    "Down (0)": number;
    "Stationary (1)": number;
    "Up (2)": number;
  };
  predicted_label: "Down" | "Stationary" | "Up";
  predicted_symbol: string;
  confidence: number;
  temperature: number;
  entropy: number;
  uncertainty_pct: number;
  raw_logits: number[];
}

export interface PredictResponse {
  scenario: string;
  scenario_description: string;
  raw_sequence: number[][];
  scaled_sequence: number[][];
  feature_names: string[];
  prediction: PredictionResult;
  engine_source?: string;
}

export interface EngineStatusResponse {
  engine: string;
  artifact_status: {
    model_exists: boolean;
    scaler_exists: boolean;
    torch_available: boolean;
    ready_for_inference: boolean;
    torch_version?: string;
  };
  model_loaded: boolean;
  scaler_loaded: boolean;
  features: string[];
  classes: string[];
}

export async function fetchEngineStatus(): Promise<EngineStatusResponse> {
  try {
    const res = await fetch("/api/status", {
      method: "GET",
      headers: { Accept: "application/json" },
      next: { revalidate: 10 },
    });
    if (res.ok) {
      const data = await res.json();
      return data;
    }
  } catch {
    // Backend offline or unavailable
  }

  // Graceful fallback status
  return {
    engine: "Offline Mode (PyTorch Backend Inactive)",
    artifact_status: {
      model_exists: true,
      scaler_exists: true,
      torch_available: false,
      ready_for_inference: false,
    },
    model_loaded: false,
    scaler_loaded: true,
    features: [
      "mid_price",
      "spread",
      "spread_bps",
      "mid_return_1s",
      "mid_return_3s",
      "mid_return_5s",
      "mid_volatility_5s",
      "bid_change_1s",
      "ask_change_1s",
      "microprice",
      "depth_imbalance",
    ],
    classes: ["Down", "Stationary", "Up"],
  };
}

export async function runPrediction(
  scenario: string,
  temperature: number = 0.85,
  customOverrides?: Record<string, number>,
  signal?: AbortSignal
): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario,
      temperature,
      custom_overrides: customOverrides || null,
    }),
    signal,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.error || `Inference service returned HTTP ${res.status}`);
  }

  return await res.json();
}
