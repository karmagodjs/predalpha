# Phase 17 — Train-Only Feature Scaling Audit Report

> [!IMPORTANT]
> **Core Architectural Proof**:
> **Scaler was fitted exclusively on training data and validation/test data were transformed using the frozen training scaler.**

- **Execution Timestamp (UTC)**: `2026-10-06T04:24:22.530464+00:00`
- **Execution Duration**: `0.34s`
- **Input Directory**: `data\clean_v2\06_sequences\new_collection`
- **Output Directory**: `data\clean_v2\07_scaled\new_collection`
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
| **Train** | `[3248, 10, 11]` | `[3248, 10, 11]` | `(3248,)` | `PASS` |
| **Validation** | `[722, 10, 11]` | `[722, 10, 11]` | `(722,)` | `PASS` |
| **Test** | `[658, 10, 11]` | `[658, 10, 11]` | `(658,)` | `PASS` |

---

## 2. Train-Learned Scaling Parameters (Fitted Strictly on X_train)

Parameters learned from 3,248 sequences across time steps ($N \times L = 32,480$ observation vectors):

| Feature | Train Mean (Center) | Train Std (Scale) | Scaled Train Mean | Scaled Train Std | Zero Variance Guard |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mid_price` | 0.500000 | 0.290551 | 3.7932e-16 | 1.000000 | NO |
| `spread` | 0.010258 | 0.003989 | 1.1809e-12 | 1.000000 | NO |
| `spread_bps` | 519.132132 | 954.508000 | 1.8838e-14 | 1.000000 | NO |
| `mid_return_1s` | 0.003528 | 0.159464 | 2.1778e-16 | 1.000000 | NO |
| `mid_return_3s` | 0.011978 | 0.301775 | -8.4035e-16 | 1.000000 | NO |
| `mid_return_5s` | 0.019889 | 0.383684 | -5.3776e-17 | 1.000000 | NO |
| `mid_volatility_5s` | 0.079608 | 0.134461 | -1.1829e-14 | 1.000000 | NO |
| `bid_change_1s` | 0.000004 | 0.037997 | 3.7105e-18 | 1.000000 | NO |
| `ask_change_1s` | -0.000004 | 0.037997 | -9.8082e-18 | 1.000000 | NO |
| `microprice` | 0.500000 | 0.291188 | 1.5269e-15 | 1.000000 | NO |
| `depth_imbalance` | 0.000000 | 0.719649 | 2.5294e-19 | 1.000000 | NO |

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

## 4. Post-Scaling Numerical Stability & Quality Audit

| Metric / Property | Train Split | Validation Split | Test Split | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **NaN Count** | 624 (0.17%) | 390 (0.49%) | 468 (0.65%) | `PASS` (Causal Warmup Only) |
| **Inf Count** | 0 | 0 | 0 | `PASS (0 Infs)` |
| **Zero-Variance Features** | 0 / 11 | N/A (Frozen Train Scaler) | N/A (Frozen Train Scaler) | `PASS` |
| **Min Value (Scaled)** | -13.1589 | -10.2640 | -7.3690 | `PASS (Stable)` |
| **Max Value (Scaled)** | 26.6978 | 15.1710 | 10.8839 | `PASS (Stable)` |
| **Class Distribution Unchanged** | `MATCH (Identical to Phase 16)` | `MATCH (Identical to Phase 16)` | `MATCH (Identical to Phase 16)` | `PASS` |

---

## 5. Artifact Hashes & Persistence

### Input Sequences (Phase 16)
- `train_sequences.parquet` (SHA-256): `c93ae311c1f6b6a12e09eb4d6375a0954ca14718c43013926e53f9be395fb86d`
- `validation_sequences.parquet` (SHA-256): `03040fc8218c088c99a96362b305d7fb69b8d11def09066f43819cf6b8ec8acd`
- `test_sequences.parquet` (SHA-256): `bc7e709468b41fb309344cafe9b2148fd021367e542f86e64fc931a03fe828c3`
- `train_sequences.npz` (SHA-256): `eb8f9823fae6e1eaa81f737143745b5643085daeeaff68b41e5523d8d142348b`
- `validation_sequences.npz` (SHA-256): `ef383211a6d7dc26e5848fd534dfd57e9f64085bc7b06cf6de9af8936401ff7b`
- `test_sequences.npz` (SHA-256): `36e2d8e2e7d5393b3e4e47449f3e189e22e22f3fe6f25f42629b5ffc0a3f9b85`

### Generated Scaled Artifacts (Phase 17)
- `train_scaled.parquet` (SHA-256): `8d625b4b0a9af0318e3220170e34eeac45d1301ee776d22546722dc05627dfd2`
- `validation_scaled.parquet` (SHA-256): `59d14857b6885519276c90e18ac26d2c9f1a5288572ab9a728e85e2c8c046994`
- `test_scaled.parquet` (SHA-256): `73f8ef374b12074213bbc213ab0bd7b29b6d8531f10a4f958744e8ce1c8f2a03`
- `train_scaled.npz` (SHA-256): `2131060963552873c0e7080440be75e06aca124ce84bc7130750637e53efc694`
- `validation_scaled.npz` (SHA-256): `864d331fdb288222fa952fda72d3f0895c7959fa7615fc5756380412c7e23ac3`
- `test_scaled.npz` (SHA-256): `d99e7e21b0f8db551666533d5a7a21aab6a504fdfce3ba779285418bccb71732`
- `scaled_production.npz` (SHA-256): `d878dce32b17d6c43e6bc1e4159d3315666bcc53b98f1b31a8f183f7f20866c3`
- `scaler_params.json` (SHA-256): `22966ed3550bc4914a4fe9a710f4cb27fe1c35d8df42208bc33134af06ccfaa9`
