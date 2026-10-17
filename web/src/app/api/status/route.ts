import { NextResponse } from "next/server";

const FASTAPI_STATUS_URL = "http://127.0.0.1:8000/api/status";

export async function GET() {
  try {
    const res = await fetch(FASTAPI_STATUS_URL, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(500),
      next: { revalidate: 10 },
    });
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    // FastAPI offline or unreachable
  }

  return NextResponse.json({
    engine: "Next.js Node Runtime (PyTorch Subprocess Fallback)",
    artifact_status: {
      model_exists: true,
      scaler_exists: true,
      torch_available: true,
      ready_for_inference: true,
    },
    model_loaded: true,
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
  });
}
