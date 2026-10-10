# Phase 17 — Train-Only Feature Scaling Audit Report

> [!IMPORTANT]
> **Core Architectural Proof**:
> **Scaler was fitted exclusively on training data and validation/test data were transformed using the frozen training scaler.**

- **Execution Timestamp (UTC)**: `2026-10-07T05:36:04.940950+00:00`
- **Execution Duration**: `2.49s`
- **Input Directory**: `data\clean_v2\06_sequences\expanded_collection`
- **Output Directory**: `data\clean_v2\07_scaled\expanded_collection`
- **Scaler Type**: `standard` (z-score standardization)
- **Fit Source**: `train` (**STRICTLY TRAIN ONLY**)
- **Feature Dimension ($D$)**: `11` causal microstructure features
- **Sequence Length ($L$)**: `10` steps
- **Final Verdict**: `**PASS**`

---

## 1. Input & Output Tensor Shape Invariants

All tensor shapes are strictly preserved before and after scaling:

| Partition | Input Shape | Scaled Output Shape | Target Labels Shape | Targets Untouched |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `[7514, 10, 11]` | `[7514, 10, 11]` | `(7514,)` | `PASS` |
| **Validation** | `[1428, 10, 11]` | `[1428, 10, 11]` | `(1428,)` | `PASS` |
| **Test** | `[1582, 10, 11]` | `[1582, 10, 11]` | `(1582,)` | `PASS` |

---

## 2. Train-Learned Scaling Parameters (Fitted Strictly on X_train)

Parameters learned from 7,514 sequences across time steps ($N \times L = 75,140$ observation vectors):

| Feature | Train Mean (Center) | Train Std (Scale) | Scaled Train Mean | Scaled Train Std | Min (Scaled) | Max (Scaled) | Zero Variance Guard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mid_price` | 0.500000 | 0.299325 | 5.6035e-16 | 1.000000 | -1.6637 | 1.6637 | NO |
| `spread` | 0.010352 | 0.003926 | 1.9011e-12 | 1.000000 | -2.3822 | 22.8361 | NO |
| `spread_bps` | 574.435690 | 1085.790641 | -5.0558e-14 | 1.000000 | -0.5198 | 13.2858 | NO |
| `mid_return_1s` | 0.001937 | 0.139058 | 8.5177e-17 | 1.000000 | -6.9388 | 30.6271 | NO |
| `mid_return_3s` | 0.006770 | 0.255785 | -7.2938e-16 | 1.000000 | -3.7947 | 20.1227 | NO |
| `mid_return_5s` | 0.011158 | 0.329714 | 1.3577e-15 | 1.000000 | -2.9571 | 15.4065 | NO |
| `mid_volatility_5s` | 0.072066 | 0.116136 | -4.0980e-15 | 1.000000 | -0.6205 | 15.8684 | NO |
| `bid_change_1s` | 0.000002 | 0.036220 | -2.2241e-18 | 1.000000 | -13.8045 | 13.2522 | NO |
| `ask_change_1s` | -0.000002 | 0.036220 | 1.6562e-18 | 1.000000 | -13.2522 | 13.8045 | NO |
| `microprice` | 0.500000 | 0.299606 | -1.5130e-18 | 1.000000 | -1.6621 | 1.6621 | NO |
| `depth_imbalance` | 0.000000 | 0.698670 | 7.4468e-19 | 1.000000 | -1.4305 | 1.4305 | NO |

---

## 3. Strict Information Leakage Audits

| Audit Verification | Requirement | Observed Verification | Verdict |
| :--- | :--- | :--- | :--- |
| **Fit Source Quarantine** | Must be fitted strictly on Train split | fit_source = `train` | `PASS` |
| **Validation Attempt Guard** | fit(X_val) must raise ValueError | `RAISED ValueError (PASSED)` | `PASS` |
| **Test Attempt Guard** | fit(X_test) must raise ValueError | `RAISED ValueError (PASSED)` | `PASS` |
| **Validation Perturbation Invariance** | Perturbing val data by 5,000x cannot alter scaler | max parameter diff = `0.00e+00` | `PASS` |
| **Test Perturbation Invariance** | Perturbing test data by 5,000x cannot alter scaler | max parameter diff = `0.00e+00` | `PASS` |
| **Deterministic Reproducibility** | Repeated fitting yields bitwise identical parameters | identical = `True` | `PASS` |
| **Inference Reloadability** | Saved scaler loads from disk and transforms identically | identical = `True` | `PASS` |

---

## 4. NaN & Explicit Causal Warm-Up Mask Accounting

| Metric / Property | Train Split | Validation Split | Test Split | Total / Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Original NaNs** | 2,418 (0.29%) | 624 (0.40%) | 546 (0.31%) | **3,588** |
| **Explicit Warm-Up Mask Cells** | 2,418 | 624 | 546 | **3,588 (Preserved)** |
| **Step 9 (Endpoint T) NaNs** | **0** | **0** | **0** | `PASS (0 NaNs at Endpoint)` |
| **Post-Imputation NaNs** | **0** | **0** | **0** | `PASS (0 NaNs Post-Imputation)` |
| **Post-Imputation Infs** | **0** | **0** | **0** | `PASS (0 Infs Post-Imputation)` |
| **Zero-Variance Features** | 0 / 11 | N/A (Frozen Train Scaler) | N/A (Frozen Train Scaler) | `PASS` |

### Timestep Breakdown of Missing Values Across Sequence Steps

| Timestep (Lookback) | Train NaNs | Validation NaNs | Test NaNs | Total NaNs | Causal Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Step 0 (T-9s) | 992 | 256 | 224 | **1,472** | Causal Warm-up |
| Step 1 (T-8s) | 620 | 160 | 140 | **920** | Causal Warm-up |
| Step 2 (T-7s) | 434 | 112 | 98 | **644** | Causal Warm-up |
| Step 3 (T-6s) | 248 | 64 | 56 | **368** | Causal Warm-up |
| Step 4 (T-5s) | 124 | 32 | 28 | **184** | Causal Warm-up |
| Step 5 (T-4s) | 0 | 0 | 0 | **0** | PASS (Post-warmup 0 NaNs) |
| Step 6 (T-3s) | 0 | 0 | 0 | **0** | PASS (Post-warmup 0 NaNs) |
| Step 7 (T-2s) | 0 | 0 | 0 | **0** | PASS (Post-warmup 0 NaNs) |
| Step 8 (T-1s) | 0 | 0 | 0 | **0** | PASS (Post-warmup 0 NaNs) |
| Step 9 (T-0s) | 0 | 0 | 0 | **0** | PASS (Endpoint 0 NaNs) |

---

## 5. Frozen Train-Scaler Validation & Test Feature Distributions

### Validation Features (Transformed with Frozen Train Scaler)

| Feature | Mean (Scaled) | Std (Scaled) | Min (Scaled) | Max (Scaled) | Warm-up NaNs | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mid_price` | 5.6388e-16 | 0.996737 | -1.6203 | 1.6203 | 0 | `PASS` |
| `spread` | 5.4822e-02 | 0.997116 | -0.0897 | 15.1942 | 0 | `PASS` |
| `spread_bps` | 7.3916e-03 | 1.055950 | -0.4355 | 5.6109 | 0 | `PASS` |
| `mid_return_1s` | -2.3948e-02 | 0.839600 | -6.8378 | 14.8252 | 16 | `PASS` |
| `mid_return_3s` | -4.6902e-02 | 0.753815 | -3.7912 | 9.0733 | 96 | `PASS` |
| `mid_return_5s` | -5.5711e-02 | 0.811469 | -2.9737 | 24.8073 | 240 | `PASS` |
| `mid_volatility_5s` | -3.9334e-03 | 0.751368 | -0.6205 | 6.9643 | 240 | `PASS` |
| `bid_change_1s` | 4.3089e-05 | 1.083054 | -17.9458 | 17.9457 | 16 | `PASS` |
| `ask_change_1s` | -4.3089e-05 | 1.083054 | -17.9457 | 17.9458 | 16 | `PASS` |
| `microprice` | 8.1941e-10 | 0.996452 | -1.6331 | 1.6331 | 0 | `PASS` |
| `depth_imbalance` | 7.0276e-08 | 0.982027 | -1.4287 | 1.4287 | 0 | `PASS` |

### Test Features (Transformed with Frozen Train Scaler)

| Feature | Mean (Scaled) | Std (Scaled) | Min (Scaled) | Max (Scaled) | Warm-up NaNs | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mid_price` | 5.5660e-16 | 0.908090 | -1.6203 | 1.6203 | 0 | `PASS` |
| `spread` | 4.3011e-02 | 0.929206 | -0.0897 | 12.6468 | 0 | `PASS` |
| `spread_bps` | -1.3118e-01 | 0.661683 | -0.4355 | 5.6109 | 0 | `PASS` |
| `mid_return_1s` | 2.3610e-03 | 1.023853 | -6.5942 | 21.4068 | 14 | `PASS` |
| `mid_return_3s` | -7.7632e-04 | 0.992791 | -3.7060 | 14.0860 | 84 | `PASS` |
| `mid_return_5s` | -3.9915e-03 | 0.958717 | -2.9332 | 11.2102 | 210 | `PASS` |
| `mid_volatility_5s` | 8.2183e-02 | 0.940979 | -0.6205 | 10.9028 | 210 | `PASS` |
| `bid_change_1s` | -8.8624e-05 | 1.204376 | -19.3263 | 19.3262 | 14 | `PASS` |
| `ask_change_1s` | 8.8624e-05 | 1.204376 | -19.3262 | 19.3263 | 14 | `PASS` |
| `microprice` | 6.2880e-18 | 0.907666 | -1.6354 | 1.6354 | 0 | `PASS` |
| `depth_imbalance` | 4.2107e-19 | 1.026509 | -1.4285 | 1.4285 | 0 | `PASS` |

---

## 6. Directional Class Distribution Preservation

| Partition | Total Sequences | UP Count (Pct) | DOWN Count (Pct) | FLAT Count (Pct) | UP/DOWN Ratio | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 7,514 | 3,050 (40.59%) | 3,050 (40.59%) | 1,414 (18.82%) | **1.0000** | `PASS` |
| **Validation** | 1,428 | 617 (43.21%) | 616 (43.14%) | 195 (13.66%) | **1.0016** | `PASS` |
| **Test** | 1,582 | 709 (44.82%) | 705 (44.56%) | 168 (10.62%) | **1.0057** | `PASS` |

---

## 7. Artifact Hashes & Persistence

### Input Sequences (Phase 16)
- `train_sequences.parquet` (SHA-256): `20d73ec929fdeb0843d08ffe197fa15da7c535eadba84eb92b119cac8910a46e`
- `validation_sequences.parquet` (SHA-256): `51eac3800dec70024b747558854694b1f98e75b14ce80215ccd657396809a787`
- `test_sequences.parquet` (SHA-256): `2c2abe39c3eb278d8a98918ae50e4611e807ae4d194850e9ce4749313f01b2e6`
- `train_sequences.npz` (SHA-256): `f49c97683c1844847770bcf7c1fe81b0e67cc35a82e86b11156b60573558569b`
- `validation_sequences.npz` (SHA-256): `caabc3d101b2307932e08de8334528615159b5b8db98ed875352b7fff5dd9cf4`
- `test_sequences.npz` (SHA-256): `cbfb24c462de2fe868f2016b9e9f00165032f793137a7b7b71fc8f0c5236ffab`

### Generated Scaled Artifacts (Phase 17)
- `train_scaled.parquet` (SHA-256): `dcf52deaf61d962d13a9ecb52274c9923bbd0aad82f4f3a8e975cc2dbbc9f358`
- `validation_scaled.parquet` (SHA-256): `539b4aa8b2a06ba1e971f8844804982823a1afcd76a9021f2d4e00c8d9e0e1d2`
- `test_scaled.parquet` (SHA-256): `ef3f22925ab1fba779bd1fb8cdce37b79be1cef6a4ae6fccf85b8f5de1adad83`
- `train_scaled.npz` (SHA-256): `ea760f237dc97573f9f06aba3215cd0c63de3a4deeb4c35f08cd5d093c780dd7`
- `validation_scaled.npz` (SHA-256): `69b97fd243bec2d2b65757ab209faa72e48c2d26a1bf6c4d1f78d6697ad62138`
- `test_scaled.npz` (SHA-256): `7b550496f89c2628ae7b6d9efa149429110fd2545153659be55237bcde35d38e`
- `scaled_production.npz` (SHA-256): `8a8b9d7a362930bedcafb792e62ca04bd4e1c4afdf75caeb63d55060c0ccd9d9`
- `warmup_masks.npz` (SHA-256): `246728f2c4caa1a2bbaa4e3ad1bcb9e04c0a67f322759c3f178abf353683593c`
- `scaler_params.json` (SHA-256): `b6fc4ae36d328d3dcafc77564870ba9fec9fcf98ecdadac3e6f570e15e3e52e7`
