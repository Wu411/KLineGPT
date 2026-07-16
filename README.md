# KLineGPT: Adapting LLM for Financial K-line Forecasting via Hierarchical Tokenization

KLineGPT is a tokenized generative forecasting framework that reformulates
multivariate OHLCV (K-line) prediction as an **autoregressive next-token modeling**
task. It hierarchically discretizes multi-channel candlestick sequences into
structured tokens via a **Temporal RQ-VAE**, transfers the sequential priors of a
frozen pre-trained LLM (GPT-2) through a lightweight adaptation (a trained
projection layer plus fine-tuning of positional embeddings `wpe` and layer
normalization `ln`), and decodes multi-scale market dynamics with a **hierarchical
cascaded head** conditioned by an **Adaptive Layer Normalization (AdaLN)** module
that re-injects historical volatility.

> This repository contains the official implementation for the KLineGPT paper.

---

## Method Overview

```
 Raw OHLCV ──▶ Causal rolling z-score ──▶ Temporal RQ-VAE ──▶ (macro, micro) tokens
                                                                      │
      market-index tokens ─▶ Gated Context Fusion ◀────────── token embeddings
                                                                      │
                                        + time-aware embeddings       ▼
                                              AdaLN (σ)  ──▶  frozen GPT-2 backbone
                                                                      │
                                        Hierarchical Cascaded Head ◀──┘
                                                                      │
                                            macro logits ─▶ micro logits (conditioned)
                                                                      ▼
                                              soft reconstruction ─▶ predictive score
                                                                      ▼
                                       daily top-K / drop-N cost-aware backtest
```

Key components:

- **Temporal RQ-VAE** (`src/model_rqvae.py`): residual vector quantization with
  `L=2` codebooks (macro / micro) that filters high-frequency noise while
  preserving structural patterns. A long-horizon variant with configurable
  temporal compression is provided in `src/model_rqvae_long.py`.
- **Modality adaptation** (`src/model_gen.py`): GPT-2 backbone with attention
  blocks frozen; only the input projection, `wpe`, and `ln` are trainable.
  Includes gated market-context fusion, time-aware embeddings, and AdaLN
  statistical injection.
- **Cascaded decoding**: the micro token is predicted conditioned on both the
  sequence context and the selected macro token.

---

## Repository Structure

```
KLineGPT/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md               # data format specification
│   ├── output_kline/           # per-stock OHLCV (not committed)
│   └── output_index_data/      # benchmark index OHLCV (not committed)
└── src/
    ├── config.py               # central configuration (edit paths here)
    ├── dataset.py              # DailyPatchDataset (tokenizer training)
    ├── model_rqvae.py          # Temporal RQ-VAE (default)
    ├── model_rqvae_long.py     # RQ-VAE variant with temporal compression
    ├── model_gen.py            # KLineGPT generative model (TimeLLM_Enhanced)
    ├── train_rq.py             # Stage 1: train the RQ-VAE tokenizer
    ├── train.py                # Stage 2: train the generative model + eval
    ├── predict.py              # inference: produce daily predictive scores
    └── backtest_sample.py      # top-K / drop-N cost-aware backtest
```

---

## Installation

```bash
git clone https://github.com/Wu411/KLineGPT.git
cd KLineGPT
python -m venv venv && source venv/bin/activate    # optional
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. In the paper, training used a single
NVIDIA A100 (80 GB).

---

## Data Preparation

Place your OHLCV data under `data/` following the format in
[`data/README.md`](data/README.md). Datasets used in the paper (obtained from
public sources) cover CSI300, CSI800, and S&P 500. All preprocessing is strictly
causal (rolling z-score with a 60-day window, `shift(1)`), and volume is log1p
scaled before normalization.

---

## Usage

All commands are run from `src/`. Edit paths and hyperparameters in
`src/config.py` first.

### Stage 1 — Train the RQ-VAE tokenizer

```bash
cd src
python train_rq.py
```

This trains the Temporal RQ-VAE, then performs a PCA-based codebook reordering
and saves `best_vqvae.pth` / `best_vqvae_sorted.pth` under `vqvae_save_dir`.

### Stage 2 — Train the generative model

```bash
python train.py
```

Loads the frozen tokenizer, initializes token embeddings from the RQ-VAE
codebook, and trains KLineGPT. It reports IC / RankIC / ICIR on the
out-of-sample split and saves `best_gen_model.pth` and per-epoch validation
scores under `gpt_save_dir/val_scores/`.

### Stage 3 — Inference

```bash
python predict.py
```

Produces daily predictive scores (CSV per date) and reports IC / RankIC / ICIR.

### Stage 4 — Backtest

```bash
python backtest_sample.py
```

Runs the daily long-only **top-K / drop-N** ranking backtest with a conservative
per-trade transaction cost and reports ARR / IR / MDD / CR, plus cumulative
return plots.

---

## Backtesting Protocol

- Daily long-only ranking; equal-weight over the **top-K** stocks.
- Turnover control: drop / buy at most **N** names per day (`(K, N) = (50, 10)`).
- Minimum holding period, next-day-open execution.
- A **conservative per-trade cost** is deducted on every buy and sell; set
  `cost_rate` in `backtest_sample.py` (paper uses `0.0015`, i.e. 0.15%).
- All reported results are **net of costs**.

---

## Key Hyperparameters (`src/config.py`)

| Parameter          | Value        | Description                          |
|--------------------|--------------|--------------------------------------|
| `window_size` (P)  | 32           | patch length                         |
| `rvq_layers` (L)   | 2            | number of RVQ codebooks              |
| `n_embeddings` (K) | [512, 512]   | codebook size per layer              |
| `norm_window` (W)  | 60           | rolling normalization window         |
| `commitment_cost`  | 0.25         | RQ-VAE commitment weight (λ)         |
| `gpt_backbone`     | gpt2         | GPT-2 small backbone                 |
| `gpt_context_len`  | 32           | context length                       |
| `pred_horizon`     | 8            | prediction horizon                   |

---

## Reproducibility Notes

- Set the random seed via `seed` in `config.py` (default 42); `train.py` seeds
  Python / NumPy / PyTorch and enables deterministic cuDNN.
- The two-stage pipeline (tokenizer → generative model) must be run in order.
- `config.py` ships with **relative** placeholder paths; adjust them to your
  environment and GPU id (`device`).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{klinegpt2026,
  title     = {KLineGPT: Adapting LLM for Financial K-line Forecasting via Hierarchical Tokenization},
  author    = {Wu, Jiaxuan and Xie, Bohong and Zhu, Peng and Cheng, Dawei and Liang, Yuqi},
  booktitle = {Proceedings of the International Conference on Advanced Data Mining and Applications (ADMA)},
  year      = {2026}
}
```

## License

Released for research purposes. Add a license file (e.g. MIT / Apache-2.0) before
public release.
