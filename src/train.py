import os
import math
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

from config import CONFIG
from model_rqvae import VQVAE_Daily
from model_gen import TimeLLM_Enhanced


# ============================================================
# 0) Utils
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_n_embeddings_list(cfg, n_layers):
    """
    - int -> [int]*n_layers
    - list/tuple -> list，并做长度对齐（截断或补齐）
    - str: 尝试 literal_eval 解析 "[1024,512,...]"，否则若为纯数字按 int 处理
    """
    n_embeddings_config = cfg["n_embeddings"]
    if isinstance(n_embeddings_config, str):
        try:
            import ast
            n_embeddings_config = ast.literal_eval(n_embeddings_config)
        except Exception:
            pass

    if isinstance(n_embeddings_config, int) or (isinstance(n_embeddings_config, str) and n_embeddings_config.isdigit()):
        num_embeddings_list = [int(n_embeddings_config)] * n_layers
    elif isinstance(n_embeddings_config, (list, tuple)):
        num_embeddings_list = list(n_embeddings_config)
    else:
        raise ValueError(f"Invalid format for n_embeddings: {n_embeddings_config}")

    if len(num_embeddings_list) != n_layers:
        if len(num_embeddings_list) > n_layers:
            num_embeddings_list = num_embeddings_list[:n_layers]
        else:
            num_embeddings_list += [num_embeddings_list[-1]] * (n_layers - len(num_embeddings_list))

    return [int(x) for x in num_embeddings_list]


def _topk_mass_from_probs(p, k):
    k_eff = int(min(max(int(k), 1), p.numel()))
    sorted_p, _ = torch.sort(p, descending=True)
    return float(sorted_p[:k_eff].sum().item()), k_eff


def compute_train_token_distribution(
    train_ds,
    n_layers,
    vocab_sizes_list,
    gpt_context_len,
    samples=200000,
    seed=123,
    topk=20,
    save_json_path=None,
):
    """
    统计训练集 token 分布（来自 RQ-VAE 编码后的真实 tokens，不是模型预测）。
    统计对象：与训练一致的 token 序列范围：tokens[:gpt_context_len+1]
    """
    rng = np.random.RandomState(int(seed))
    N = len(train_ds.indices)

    if samples is None or int(samples) <= 0 or int(samples) >= N:
        sel = np.arange(N, dtype=np.int64)
    else:
        sel = rng.choice(N, size=int(samples), replace=False)

    counts = [torch.zeros((vocab_sizes_list[i],), dtype=torch.long) for i in range(n_layers)]
    total_tokens = [0 for _ in range(n_layers)]

    window_size = train_ds.window_size
    num_patches_total = train_ds.num_patches_total

    for idx in tqdm(sel, desc="[*] Computing Train Token Dist", leave=False):
        seq_idx, p0, _ = train_ds.indices[int(idx)]
        patch_indices = [p0 + k * window_size for k in range(num_patches_total)]
        tok = train_ds.stock_tokens[seq_idx][patch_indices]          # [P, latent_len, L]
        tok = tok.reshape(-1, tok.shape[-1])                        # [P*latent_len, L]

        if tok.shape[0] <= gpt_context_len:
            continue

        tok = tok[: gpt_context_len + 1, :]                         # 与训练 target_ids 范围对齐

        for li in range(n_layers):
            ids = tok[:, li].astype(np.int64)
            ids_t = torch.from_numpy(ids)

            mn = int(ids_t.min().item())
            mx = int(ids_t.max().item())
            if mn < 0 or mx >= vocab_sizes_list[li]:
                raise RuntimeError(
                    f"Token out of range at layer {li}: min={mn}, max={mx}, K={vocab_sizes_list[li]}"
                )

            counts[li] += torch.bincount(ids_t, minlength=vocab_sizes_list[li])
            total_tokens[li] += int(ids_t.numel())

    metrics = {}
    print("\n" + "=" * 50)
    print("[Train Token Distribution Summary]")
    for li in range(n_layers):
        c = counts[li].float()
        denom = c.sum() + 1e-9
        p = c / denom

        entropy = -torch.sum(p * torch.log(p + 1e-9))
        ppl = float(torch.exp(entropy).item())
        usage = float((counts[li] > 0).float().mean().item())
        topk_mass, k_eff = _topk_mass_from_probs(p, topk)
        top5 = torch.argsort(p, descending=True)[:5].cpu().numpy().tolist()

        metrics[f"L{li}"] = {
            "K": int(vocab_sizes_list[li]),
            "Tokens": int(total_tokens[li]),
            "PPL": ppl,
            "Usage": usage,
            f"Top{k_eff}_Mass": topk_mass,
            "Top5": top5,
        }

        print(
            f"  Layer {li}: K={vocab_sizes_list[li]} | Tokens={total_tokens[li]} | "
            f"PPL={ppl:.2f} | Usage={usage*100:.1f}% | Top{k_eff}_Mass={topk_mass*100:.1f}% | Top5={top5}"
        )
    print("=" * 50 + "\n")

    if save_json_path is not None:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[*] Saved train token distribution to: {save_json_path}")

    return metrics


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.clone()
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.clone()
        logits[indices_to_remove] = filter_value

    return logits


def visualize_best_worst(
    pred_prices,
    target_token_prices,
    true_prices,
    dates,
    codes,
    per_sample_losses,
    save_dir,
    epoch,
    count=16,
    plot_mode="return",
):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = pred_prices.shape[0]

    real_count = min(batch_size, count)
    if real_count % 2 != 0:
        real_count -= 1
    if real_count <= 0:
        return

    half_count = real_count // 2

    if isinstance(per_sample_losses, torch.Tensor):
        per_sample_losses = per_sample_losses.detach().cpu().numpy()

    sorted_indices = np.argsort(per_sample_losses)
    best_indices = sorted_indices[:half_count]
    worst_indices = sorted_indices[-half_count:]
    viz_indices = np.concatenate([best_indices, worst_indices])

    cols = half_count
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 8), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols == 1:
        axes = np.expand_dims(axes, 1)

    axes = axes.flatten()

    for i, idx in enumerate(viz_indices):
        ax = axes[i]

        pred = pred_prices[idx].detach().cpu().numpy()
        target_recon = target_token_prices[idx].detach().cpu().numpy()
        true_raw = true_prices[idx].detach().cpu().numpy()

        is_best = i < half_count
        group_name = "BEST" if is_best else "WORST"
        title_color = "darkgreen" if is_best else "darkred"

        if plot_mode == "return":
            base = true_raw[0] + 1e-8
            plot_true = (true_raw - true_raw[0]) / base
            plot_pred = (pred - pred[0]) / (pred[0] + 1e-8)
            plot_target = (target_recon - target_recon[0]) / (target_recon[0] + 1e-8)
            ylabel = "Rel Return"
        else:
            plot_true = true_raw
            plot_pred = pred
            plot_target = target_recon
            ylabel = "Price"

        base_calc = true_raw[0] + 1e-8
        norm_pred_calc = (pred - pred[0]) / (pred[0] + 1e-8)
        norm_true_calc = (true_raw - true_raw[0]) / base_calc
        vis_mse = ((norm_pred_calc - norm_true_calc) ** 2).mean()

        same_sign = np.sign(norm_pred_calc[-1]) == np.sign(norm_true_calc[-1])
        sign_str = "RIGHT" if same_sign else "WRONG"

        x_axis = range(len(true_raw))
        dt_item = dates[idx].item()
        try:
            dt_str = pd.to_datetime(dt_item).strftime("%Y-%m-%d")
        except Exception:
            dt_str = str(dt_item)

        ax.plot(x_axis, plot_true, label="True", linewidth=2.0, alpha=0.6)
        ax.plot(x_axis, plot_target, label="VQ", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.plot(x_axis, plot_pred, label="Pred", linestyle="-", linewidth=2.0)

        ax.set_title(
            f"[{group_name}] MSE:{vis_mse:.4f} | {sign_str}\n{codes[idx]} | {dt_str}",
            fontsize=9,
            color=title_color,
            fontweight="bold",
        )

        if i % cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i == 0 or i == half_count:
            ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_{plot_mode}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 1) Dataset
# ============================================================
class RVQSequenceDataset(Dataset):
    def __init__(self, df, vqvae_model, config, device="cuda", min_decision_date=None, seq_stride_days=1):
        self.window_size = int(config["window_size"])
        self.norm_window = int(config.get("norm_window", 60))
        self.eps = float(config.get("norm_eps", 1e-6))
        self.clip_value = float(config.get("norm_clip", 10.0))
        self.encode_batch_size = int(config.get("encode_batch_size", 512))
        self.gpt_context_len = int(config["gpt_context_len"])
        self.seq_stride_days = int(seq_stride_days)
        self.min_decision_date = pd.to_datetime(min_decision_date) if min_decision_date is not None else None
        self.device = device
        self.use_market_context = config.get("use_market_context", False)

        self.indices = []
        self.stock_tokens = []
        self.stock_close_mu = []
        self.stock_close_sd = []
        self.stock_true_next_ohlc = []
        self.stock_kdcode = []
        self.stock_last_close = []
        self.patch_dates = []
        self.time_features = []

        vqvae_model.eval().to(device)

        # Step A: Market Index
        self.date_to_index_token = {}
        if self.use_market_context:
            index_path = config.get("index_data_path", "")
            if os.path.exists(index_path):
                print(f"[*] Processing Market Index from {index_path}...")
                df_index = pd.read_csv(index_path)
                time_col = "date" if "date" in df_index.columns else "dt"
                df_index["dt"] = pd.to_datetime(df_index[time_col].astype(str))
                df_index = df_index.sort_values("dt").reset_index(drop=True)

                if min_decision_date is None:
                    max_stock_dt = pd.to_datetime(df["dt"].max())
                    df_index = df_index[df_index["dt"] <= max_stock_dt].reset_index(drop=True)
                    print(f"    [-] Index Data Clipped to {max_stock_dt} (Train Mode)")

                self._process_and_cache_index(df_index, vqvae_model)
            else:
                print(f"[!] Warning: Index path {index_path} not found. Context will be zero-padded.")

        # Step B: Stock Data
        df = df.copy().sort_values(["kdcode", "dt"]).reset_index(drop=True)
        groups = df.groupby("kdcode").indices

        o_all = df["open"].to_numpy(np.float32)
        h_all = df["high"].to_numpy(np.float32)
        l_all = df["low"].to_numpy(np.float32)
        c_all = df["close"].to_numpy(np.float32)
        v_all = df["volume"].to_numpy(np.float32)
        dt_all = df["dt"].to_numpy("datetime64[ns]")

        self.latent_len = None

        print("[*] Encoding stocks & Preparing Data...")
        with torch.no_grad():
            for kdcode, idx_array in tqdm(groups.items()):
                idx_array = np.asarray(idx_array, dtype=np.int64)
                n = len(idx_array)
                if n <= (self.norm_window + self.window_size + 6):
                    continue

                o_series = pd.Series(o_all[idx_array])
                h_series = pd.Series(h_all[idx_array])
                l_series = pd.Series(l_all[idx_array])
                c_series = pd.Series(c_all[idx_array])
                v_series = pd.Series(np.log1p(np.maximum(v_all[idx_array], 0.0)).astype(np.float32))

                o_mu = o_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                o_sd = o_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                h_mu = h_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                h_sd = h_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                l_mu = l_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                l_sd = l_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                c_mu = c_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                c_sd = c_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                v_mu = v_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                v_sd = v_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)

                valid_mask = (
                    np.isfinite(o_mu)
                    & (o_sd > 1e-8)
                    & np.isfinite(h_mu)
                    & (h_sd > 1e-8)
                    & np.isfinite(l_mu)
                    & (l_sd > 1e-8)
                    & np.isfinite(c_mu)
                    & (c_sd > 1e-8)
                    & np.isfinite(v_mu)
                    & (v_sd > 1e-8)
                )
                valid_starts = np.where(valid_mask)[0]
                valid_starts = valid_starts[(valid_starts >= self.norm_window) & (valid_starts <= n - self.window_size - 5)]
                if len(valid_starts) == 0:
                    continue

                stock_codes_list = []
                for b0 in range(0, len(valid_starts), self.encode_batch_size):
                    batch = []
                    for s in valid_starts[b0: b0 + self.encode_batch_size]:
                        mu = np.array([o_mu[s], h_mu[s], l_mu[s], c_mu[s], v_mu[s]], dtype=np.float32)
                        sd = np.array([max(x, self.eps) for x in [o_sd[s], h_sd[s], l_sd[s], c_sd[s], v_sd[s]]], dtype=np.float32)

                        o_patch = o_all[idx_array[s: s + self.window_size]]
                        h_patch = h_all[idx_array[s: s + self.window_size]]
                        l_patch = l_all[idx_array[s: s + self.window_size]]
                        c_patch = c_all[idx_array[s: s + self.window_size]]
                        v_patch = np.log1p(np.maximum(v_all[idx_array[s: s + self.window_size]], 0.0)).astype(np.float32)

                        norm_o = (o_patch - mu[0]) / sd[0]
                        norm_h = (h_patch - mu[1]) / sd[1]
                        norm_l = (l_patch - mu[2]) / sd[2]
                        norm_c = (c_patch - mu[3]) / sd[3]
                        norm_v = (v_patch - mu[4]) / sd[4]

                        patch = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=0)
                        patch = np.clip(patch, -self.clip_value, self.clip_value)
                        batch.append(patch)

                    x_in = torch.tensor(np.array(batch), dtype=torch.float32).to(device)
                    encoded = vqvae_model.encode(x_in).cpu()
                    stock_codes_list.append(encoded)

                stock_codes = torch.cat(stock_codes_list, dim=0).numpy().astype(np.int64)
                if self.latent_len is None:
                    self.latent_len = stock_codes.shape[1]
                    self.num_patches_context = int(np.ceil(self.gpt_context_len / self.latent_len))
                    self.num_patches_total = self.num_patches_context + 1
                self.stock_tokens.append(stock_codes)

                mu_arr = np.stack([o_mu, h_mu, l_mu, c_mu, v_mu], axis=1)
                sd_arr = np.stack([o_sd, h_sd, l_sd, c_sd, v_sd], axis=1)
                self.stock_close_mu.append(mu_arr[valid_starts])
                self.stock_close_sd.append(sd_arr[valid_starts])

                current_ohlc_chunk = []
                current_last_close_chunk = []
                current_patch_dates = []
                current_time_feats = []
                for s in valid_starts:
                    current_ohlc_chunk.append(c_all[idx_array[s: s + self.window_size]])
                    current_last_close_chunk.append(c_all[idx_array[s - 1]])
                    dt_ns = dt_all[idx_array[s - 1]]
                    ts = pd.Timestamp(dt_ns)
                    current_patch_dates.append(ts.value)
                    current_time_feats.append([ts.day, ts.month, ts.dayofweek])

                self.stock_true_next_ohlc.append(current_ohlc_chunk)
                self.stock_last_close.append(current_last_close_chunk)
                self.stock_kdcode.append(str(kdcode))
                self.patch_dates.append(current_patch_dates)
                self.time_features.append(current_time_feats)

                n_patches = len(valid_starts)
                max_start_idx = n_patches - 1 - (self.num_patches_total - 1) * self.window_size
                if max_start_idx < 0:
                    continue

                for p0 in range(0, max_start_idx + 1, self.seq_stride_days):
                    target_relative_idx = (self.num_patches_total - 1) * self.window_size
                    target_patch_start_idx_in_valid = p0 + target_relative_idx
                    decision_dt_val = current_patch_dates[target_patch_start_idx_in_valid]
                    if self.min_decision_date and pd.Timestamp(decision_dt_val) < self.min_decision_date:
                        continue
                    self.indices.append((len(self.stock_tokens) - 1, p0, decision_dt_val))

    def _process_and_cache_index(self, df_index, model):
        o = df_index["open"].to_numpy(np.float32)
        h = df_index["high"].to_numpy(np.float32)
        l = df_index["low"].to_numpy(np.float32)
        c = df_index["close"].to_numpy(np.float32)
        v = np.log1p(np.maximum(df_index["volume"].to_numpy(np.float32), 0.0)).astype(np.float32)
        dts = df_index["dt"]

        o_s = pd.Series(o)
        o_mu = o_s.rolling(self.norm_window).mean().shift(1).to_numpy()
        o_sd = o_s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy()
        h_s = pd.Series(h)
        h_mu = h_s.rolling(self.norm_window).mean().shift(1).to_numpy()
        h_sd = h_s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy()
        l_s = pd.Series(l)
        l_mu = l_s.rolling(self.norm_window).mean().shift(1).to_numpy()
        l_sd = l_s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy()
        c_s = pd.Series(c)
        c_mu = c_s.rolling(self.norm_window).mean().shift(1).to_numpy()
        c_sd = c_s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy()
        v_s = pd.Series(v)
        v_mu = v_s.rolling(self.norm_window).mean().shift(1).to_numpy()
        v_sd = v_s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy()

        valid_mask = np.isfinite(o_mu) & (o_sd > 1e-8)
        valid_indices = np.where(valid_mask)[0]
        starts = valid_indices[valid_indices <= len(df_index) - self.window_size]

        batch_size = 512
        count = 0
        for b0 in range(0, len(starts), batch_size):
            batch_starts = starts[b0: b0 + batch_size]
            batch_patches = []
            batch_keys = []
            for s in batch_starts:
                mu = np.array([o_mu[s], h_mu[s], l_mu[s], c_mu[s], v_mu[s]])
                sd = np.array([o_sd[s], h_sd[s], l_sd[s], c_sd[s], v_sd[s]])
                sd = np.maximum(sd, self.eps)

                patch_o = (o[s: s + self.window_size] - mu[0]) / sd[0]
                patch_h = (h[s: s + self.window_size] - mu[1]) / sd[1]
                patch_l = (l[s: s + self.window_size] - mu[2]) / sd[2]
                patch_c = (c[s: s + self.window_size] - mu[3]) / sd[3]
                patch_v = (v[s: s + self.window_size] - mu[4]) / sd[4]

                patch = np.stack([patch_o, patch_h, patch_l, patch_c, patch_v], axis=0)
                batch_patches.append(np.clip(patch, -self.clip_value, self.clip_value))
                key_date = dts.iloc[s - 1].value
                batch_keys.append(key_date)

            x_in = torch.tensor(np.array(batch_patches), dtype=torch.float32).to(self.device)
            codes = model.encode(x_in).cpu().numpy().astype(np.int64)
            if len(codes.shape) == 2:
                codes = codes[..., None]
            for k, code in zip(batch_keys, codes):
                self.date_to_index_token[k] = code
            count += len(batch_keys)
        print(f"[*] Indexed {count} market context patches.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        seq_idx, p0, date_ns = self.indices[i]
        patch_indices = [p0 + k * self.window_size for k in range(self.num_patches_total)]

        tokens = self.stock_tokens[seq_idx][patch_indices]
        tokens = tokens.reshape(-1, tokens.shape[-1])

        target_valid_start_idx = patch_indices[-1]
        mu = self.stock_close_mu[seq_idx][target_valid_start_idx]
        sd = self.stock_close_sd[seq_idx][target_valid_start_idx]
        true_close_patch = self.stock_true_next_ohlc[seq_idx][target_valid_start_idx]
        kdcode = self.stock_kdcode[seq_idx]
        last_close = self.stock_last_close[seq_idx][target_valid_start_idx]

        context_patch_indices = patch_indices[:-1]
        n_layers = tokens.shape[-1]
        latent_len = self.latent_len
        gpt_context_len = self.gpt_context_len

        time_feats_list = []
        market_tokens_list = []
        dummy_market = np.zeros((latent_len, n_layers), dtype=np.int64)

        for p_idx in context_patch_indices:
            t_f = self.time_features[seq_idx][p_idx]
            time_feats_list.append(t_f)

            if self.use_market_context:
                ts_val = self.patch_dates[seq_idx][p_idx]
                m_token = self.date_to_index_token.get(ts_val, dummy_market)
                if m_token.shape != dummy_market.shape:
                    m_token = dummy_market
            else:
                m_token = dummy_market

            market_tokens_list.append(m_token)

        time_feats = torch.as_tensor(time_feats_list, dtype=torch.long)
        time_feats = time_feats.repeat_interleave(latent_len, dim=0)

        market_tokens = np.stack(market_tokens_list, axis=0).astype(np.int64)
        market_tokens = torch.from_numpy(market_tokens).long().reshape(-1, n_layers)

        if time_feats.size(0) >= gpt_context_len:
            time_feats = time_feats[:gpt_context_len]
        else:
            pad = gpt_context_len - time_feats.size(0)
            time_feats = torch.cat([time_feats, torch.zeros(pad, 3, dtype=torch.long)], dim=0)

        if market_tokens.size(0) >= gpt_context_len:
            market_tokens = market_tokens[:gpt_context_len]
        else:
            pad = gpt_context_len - market_tokens.size(0)
            market_tokens = torch.cat([market_tokens, torch.zeros(pad, n_layers, dtype=torch.long)], dim=0)

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(mu, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32),
            torch.tensor(date_ns, dtype=torch.long),
            torch.tensor(true_close_patch, dtype=torch.float32),
            torch.tensor(last_close, dtype=torch.float32),
            kdcode,
            time_feats,
            market_tokens,
        )


# ============================================================
# 2) Metric Recorder
# ============================================================
class KronosMetricRecorder:
    def __init__(self):
        self.records = {"date": [], "pred": [], "true": []}

    def update(self, batch_pred_ret, batch_true_ret, batch_dates):
        if isinstance(batch_pred_ret, torch.Tensor):
            batch_pred_ret = batch_pred_ret.detach().cpu().numpy()
        if isinstance(batch_true_ret, torch.Tensor):
            batch_true_ret = batch_true_ret.detach().cpu().numpy()
        if isinstance(batch_dates, torch.Tensor):
            batch_dates = batch_dates.detach().cpu().numpy()
        self.records["date"].append(batch_dates)
        self.records["pred"].append(batch_pred_ret)
        self.records["true"].append(batch_true_ret)

    def compute(self):
        if not self.records["date"]:
            return {"return_ic": 0.0, "return_rankic": 0.0, "return_icir": 0.0}
        all_dates = np.concatenate(self.records["date"])
        all_preds = np.concatenate(self.records["pred"])
        all_trues = np.concatenate(self.records["true"])
        df = pd.DataFrame({"dt": all_dates, "pred": all_preds, "true": all_trues})

        def calc_daily_metrics(g):
            if len(g) < 2 or g["pred"].std() == 0 or g["true"].std() == 0:
                return pd.Series({"rank_ic": np.nan, "ic": np.nan})
            rank_ic = spearmanr(g["pred"], g["true"])[0]
            ic = pearsonr(g["pred"], g["true"])[0]
            return pd.Series({"rank_ic": rank_ic, "ic": ic})

        daily_metrics = df.groupby("dt").apply(calc_daily_metrics, include_groups=False)
        mean_rank_ic = daily_metrics["rank_ic"].mean()
        std_rank_ic = daily_metrics["rank_ic"].std()
        mean_ic = daily_metrics["ic"].mean()
        icir = mean_rank_ic / (std_rank_ic + 1e-9)
        return {"return_ic": mean_ic, "return_rankic": mean_rank_ic, "return_icir": icir}


class ReconMetricRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mse_sum = 0.0
        self.count = 0

    def update(self, batch_mse, n=1):
        self.mse_sum += float(batch_mse) * int(n)
        self.count += int(n)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.mse_sum / self.count


class TokenCollapseRecorderVarVocab:
    """
    变长码本版本：每层 vocab_size 不同（用于监控模型预测是否坍塌）
    """
    def __init__(self, n_layers, vocab_sizes_list, device="cpu", monitor_top_k=20):
        if len(vocab_sizes_list) != n_layers:
            raise ValueError("vocab_sizes_list length must equal n_layers")
        self.n_layers = n_layers
        self.vocab_sizes_list = [int(x) for x in vocab_sizes_list]
        self.monitor_top_k = int(monitor_top_k)
        self.token_counts = [
            torch.zeros((self.vocab_sizes_list[i],), dtype=torch.long, device=device)
            for i in range(n_layers)
        ]
        self.total_samples = 0

    def update(self, logits_list):
        batch_size = logits_list[0].size(0)
        self.total_samples += int(batch_size)
        for i, logits in enumerate(logits_list):
            if logits.dim() > 2:
                logits = logits.reshape(-1, logits.size(-1))
            pred_ids = torch.argmax(logits, dim=-1)
            K = self.vocab_sizes_list[i]
            counts = torch.bincount(pred_ids, minlength=K)
            self.token_counts[i] += counts.to(self.token_counts[i].device)

    def compute(self, verbose=True):
        metrics = {}
        for i in range(self.n_layers):
            counts = self.token_counts[i].float()
            denom = counts.sum() + 1e-9
            p = counts / denom

            entropy = -torch.sum(p * torch.log(p + 1e-9))
            perplexity = torch.exp(entropy).item()

            active_mask = (self.token_counts[i] > 0).float()
            usage_ratio = active_mask.mean().item()

            sorted_probs, sorted_indices = torch.sort(p, descending=True)
            k_eff = min(self.monitor_top_k, sorted_probs.numel())
            top_k_mass = sorted_probs[:k_eff].sum().item()
            top_k_indices = sorted_indices[:5].cpu().numpy().tolist()

            metrics[f"L{i}_PPL"] = perplexity
            metrics[f"L{i}_Usage"] = usage_ratio
            metrics[f"L{i}_Top{k_eff}_Mass"] = top_k_mass

            if verbose:
                status = "OK"
                if top_k_mass > 0.90 or usage_ratio < 0.01:
                    status = "⚠️ COLLAPSE"
                elif top_k_mass > 0.70:
                    status = "WARNING"

                print(
                    f"  Layer {i}: K={self.vocab_sizes_list[i]} | "
                    f"PPL={perplexity:.2f} | Usage={usage_ratio*100:.1f}% | "
                    f"Top{k_eff}_Mass={top_k_mass*100:.1f}% | "
                    f"Dominant IDs:{top_k_indices} | {status}"
                )

        return metrics


# ============================================================
# 3) Run Epoch
# ============================================================
def _build_ce_layer_weights(cfg, n_layers):
    if "ce_layer_weights" in cfg and cfg["ce_layer_weights"] is not None:
        w = cfg["ce_layer_weights"]
        if isinstance(w, str):
            try:
                import ast
                w = ast.literal_eval(w)
            except Exception:
                pass
        if isinstance(w, (list, tuple)):
            w = list(w)
            if len(w) != n_layers:
                if len(w) > n_layers:
                    w = w[:n_layers]
                else:
                    w += [w[-1]] * (n_layers - len(w))
            return [float(x) for x in w]
        if isinstance(w, (int, float)):
            return [float(w)] * n_layers

    if "ce_layer_weight" in cfg and cfg["ce_layer_weight"] is not None:
        return [float(cfg["ce_layer_weight"])] * n_layers

    return [1.0] * n_layers


# ============================================================
# 3) Run Epoch
# ============================================================
def run_epoch(mode, model, vqvae, loader, optimizer, device, epoch_idx, collect_signal=False, vis_save_dir=None):
    is_train = (mode == "train")
    model.train() if is_train else model.eval()
    vqvae.eval()

    rec_returns = KronosMetricRecorder()
    rec_recon = ReconMetricRecorder()

    # 原有：全层 oracle（所有残差层求和）
    rec_oracle_full = KronosMetricRecorder()

    # 新增：逐层 oracle（前缀 0..i 的重构）
    n_layers = model.n_rvq_layers
    rec_oracle_prefix = [KronosMetricRecorder() for _ in range(n_layers)]

    loss_sum = 0.0
    loss_ce_sum = 0.0

    ce_layer_sums = [0.0] * n_layers
    acc_layer_sums = [0.0] * n_layers
    n_last_sum = 0

    ce_layer_weights = _build_ce_layer_weights(CONFIG, n_layers)

    token_monitor = None
    if not is_train:
        vocab_sizes_list = parse_n_embeddings_list(CONFIG, n_layers)
        token_monitor = TokenCollapseRecorderVarVocab(
            n_layers=n_layers,
            vocab_sizes_list=vocab_sizes_list,
            device=device,
            monitor_top_k=20
        )

    pbar = tqdm(loader, desc=f"Ep {epoch_idx} [{mode}]", leave=False)

    vis_done = False
    target_vis_batch_idx = -1
    if (not is_train) and (vis_save_dir is not None):
        target_vis_batch_idx = np.random.randint(0, len(loader))

    all_signals, all_dates, all_codes, all_true_prices = [], [], [], []

    for batch_i, (tokens, mu, sd, date_ns, true_close_patch, last_close, kdcode, time_feats, market_tokens) in enumerate(pbar):
        tokens = tokens.to(device)
        mu = mu.to(device)
        sd = sd.to(device)
        true_close_patch = true_close_patch.to(device)
        last_close = last_close.to(device)
        time_feats = time_feats.to(device)
        market_tokens = market_tokens.to(device)

        stats_feats = sd

        gpt_context_len = int(CONFIG["gpt_context_len"])
        H_pred = int(CONFIG.get("pred_horizon", 8))

        if tokens.shape[1] <= gpt_context_len:
            continue

        context_tokens = tokens[:, :gpt_context_len, :]
        full_target_ids = tokens[:, 1: gpt_context_len + 1, :]
        true_close_patch = true_close_patch[:, :H_pred]

        curr_market = market_tokens[:, :gpt_context_len, :] if market_tokens is not None else None
        curr_time = time_feats[:, :gpt_context_len, :] if time_feats is not None else None

        # ==================== Train ====================
        if is_train:
            optimizer.zero_grad()

            logits_list = model(
                context_tokens,
                market_ids=curr_market,
                time_feats=curr_time,
                target_ids=full_target_ids,
                stats_feats=stats_feats,
            )

            total_loss = torch.tensor(0.0, device=device)
            batch_ce = 0.0

            for i in range(n_layers):
                layer_logits = logits_list[i]
                layer_target = full_target_ids[:, :, i]
                ce = F.cross_entropy(
                    layer_logits.reshape(-1, layer_logits.size(-1)),
                    layer_target.reshape(-1),
                    label_smoothing=0.1,
                )
                layer_w = float(ce_layer_weights[i])
                ce_weighted = ce * layer_w
                total_loss += ce_weighted
                batch_ce += float(ce_weighted.item())

            with torch.no_grad():
                bs = int(full_target_ids.size(0))
                n_last_sum += bs
                for i in range(n_layers):
                    last_logits = logits_list[i][:, -1, :]
                    last_target = full_target_ids[:, -1, i]
                    ce_raw = F.cross_entropy(last_logits, last_target)
                    pred_ids = torch.argmax(last_logits, dim=-1)
                    acc = (pred_ids == last_target).float().mean().item()
                    ce_layer_sums[i] += float(ce_raw.item()) * bs
                    acc_layer_sums[i] += float(acc) * bs

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += float(total_loss.item())
            loss_ce_sum += float(batch_ce)

        # ==================== Eval ====================
        else:
            with torch.no_grad():
                # 1) CE forward (teacher forcing)
                logits_tf_list = model(
                    context_tokens,
                    market_ids=curr_market,
                    time_feats=curr_time,
                    target_ids=full_target_ids,
                    stats_feats=stats_feats,
                )

                batch_ce = 0.0
                total_batch_loss = 0.0
                for i in range(n_layers):
                    layer_logits = logits_tf_list[i]
                    layer_target = full_target_ids[:, :, i]
                    ce = F.cross_entropy(
                        layer_logits.reshape(-1, layer_logits.size(-1)),
                        layer_target.reshape(-1),
                    )
                    layer_w = float(ce_layer_weights[i])
                    total_batch_loss += float(ce.item()) * layer_w
                    batch_ce += float(ce.item()) * layer_w

                bs = int(full_target_ids.size(0))
                n_last_sum += bs
                for i in range(n_layers):
                    last_logits = logits_tf_list[i][:, -1, :]
                    last_target = full_target_ids[:, -1, i]
                    ce_raw = F.cross_entropy(last_logits, last_target)
                    pred_ids = torch.argmax(last_logits, dim=-1)
                    acc = (pred_ids == last_target).float().mean().item()
                    ce_layer_sums[i] += float(ce_raw.item()) * bs
                    acc_layer_sums[i] += float(acc) * bs

                # 2) Inference forward
                logits_inf_list = model(
                    context_tokens,
                    market_ids=curr_market,
                    time_feats=curr_time,
                    target_ids=None,
                    stats_feats=stats_feats,
                )

                if token_monitor is not None:
                    token_monitor.update(logits_inf_list)

                last_step_logits_list = [l[:, -1, :] for l in logits_inf_list]

                mu_c = mu[:, 3].unsqueeze(1)
                sd_c = sd[:, 3].unsqueeze(1)

                # Soft Reconstruction (model pred)
                z_soft_sum = 0.0
                for i in range(n_layers):
                    probs = F.softmax(last_step_logits_list[i], dim=-1)
                    codebook_w = vqvae.rvq.layers[i]._embedding.weight
                    z_soft_sum = z_soft_sum + torch.matmul(probs, codebook_w)

                z_soft = z_soft_sum.unsqueeze(-1)
                recon_norm = vqvae.decoder(z_soft)
                pred_close_norm = recon_norm[:, 3, :H_pred]
                pred_price = pred_close_norm * sd_c + mu_c

                # Recon MSE
                true_norm_sub = (true_close_patch - mu_c) / (sd_c + 1e-8)
                avg_close_norm = (pred_price - mu_c) / (sd_c + 1e-8)
                true_norm_sub = torch.clamp(true_norm_sub, -10.0, 10.0)
                avg_close_norm = torch.clamp(avg_close_norm, -10.0, 10.0)
                batch_recon_mse = float(F.mse_loss(avg_close_norm, true_norm_sub).item())
                rec_recon.update(batch_recon_mse, tokens.size(0))

                # Signals
                pred_mean_signal = (pred_price.mean(dim=1) - last_close) / (last_close + 1e-8)
                true_mean_label = (true_close_patch.mean(dim=1) - last_close) / (last_close + 1e-8)

                # ========= Oracle Signals (逐层前缀) =========
                z_prefix = 0.0
                oracle_signal_prefix = []
                for i in range(n_layers):
                    oracle_ids_i = full_target_ids[:, -1, i]
                    emb_i = vqvae.rvq.layers[i]._embedding(oracle_ids_i)  # [B, emb_dim]
                    z_prefix = z_prefix + emb_i
                    z_i = z_prefix.unsqueeze(-1)                          # [B, emb_dim, 1]
                    oracle_recon_norm_i = vqvae.decoder(z_i)
                    oracle_close_norm_i = oracle_recon_norm_i[:, 3, :H_pred]
                    oracle_price_i = oracle_close_norm_i * sd_c + mu_c
                    oracle_signal_i = (oracle_price_i.mean(dim=1) - last_close) / (last_close + 1e-8)
                    oracle_signal_prefix.append(oracle_signal_i)

                oracle_signal_full = oracle_signal_prefix[-1]

                loss_sum += float(total_batch_loss)
                loss_ce_sum += float(batch_ce)

                # 更新 IC/RankIC recorder
                rec_returns.update(pred_mean_signal, true_mean_label, date_ns)

                rec_oracle_full.update(oracle_signal_full, true_mean_label, date_ns)
                for i in range(n_layers):
                    rec_oracle_prefix[i].update(oracle_signal_prefix[i], true_mean_label, date_ns)

                should_visualize = (vis_save_dir is not None and (not vis_done)) and (batch_i == target_vis_batch_idx)
                if should_visualize:
                    z_full = z_prefix.unsqueeze(-1)
                    oracle_recon_norm = vqvae.decoder(z_full)
                    oracle_close_norm = oracle_recon_norm[:, 3, :H_pred]
                    oracle_price = oracle_close_norm * sd_c + mu_c

                    visualize_best_worst(
                        pred_price,
                        oracle_price,
                        true_close_patch,
                        date_ns,
                        kdcode,
                        (pred_mean_signal - true_mean_label) ** 2,
                        vis_save_dir,
                        epoch_idx,
                        count=16,
                        plot_mode="price",
                    )
                    vis_done = True

                if collect_signal:
                    all_signals.append(pred_mean_signal.cpu().numpy())
                    all_dates.extend(date_ns.cpu().numpy())
                    all_codes.extend(kdcode)
                    all_true_prices.append(true_close_patch.mean(dim=1).cpu().numpy())

        denom = max(n_last_sum, 1)
        layer_metrics = {}
        for i in range(n_layers):
            layer_metrics[f"ce_l{i}"] = ce_layer_sums[i] / denom
            layer_metrics[f"acc_l{i}"] = acc_layer_sums[i] / denom

        pbar_dict = {"Loss": f"{loss_sum/(batch_i+1):.3f}"}
        for i in range(n_layers):
            pbar_dict[f"L{i}ce"] = f"{layer_metrics[f'ce_l{i}']:.2f}"
            pbar_dict[f"L{i}Acc"] = f"{layer_metrics[f'acc_l{i}']*100:.2f}%"
        pbar.set_postfix(pbar_dict)

    metrics = rec_returns.compute()
    oracle_full_metrics = rec_oracle_full.compute()
    oracle_prefix_metrics = [rec_oracle_prefix[i].compute() for i in range(n_layers)]

    avg_loss = loss_sum / max(len(loader), 1)
    avg_ce_loss = loss_ce_sum / max(len(loader), 1)
    avg_trend_loss = 0.0  # 已删除 trend_loss
    avg_recon_error = rec_recon.compute()

    denom = max(n_last_sum, 1)
    final_layer_metrics = []
    for i in range(n_layers):
        ce_avg = ce_layer_sums[i] / denom
        acc_avg = acc_layer_sums[i] / denom
        final_layer_metrics.append((ce_avg, acc_avg))

    metrics["oracle_rankic"] = oracle_full_metrics.get("return_rankic", 0.0)
    metrics["oracle_ic"] = oracle_full_metrics.get("return_ic", 0.0)
    metrics["oracle_icir"] = oracle_full_metrics.get("return_icir", 0.0)

    for i in range(n_layers):
        metrics[f"oracle_l{i}_rankic"] = oracle_prefix_metrics[i].get("return_rankic", 0.0)
        metrics[f"oracle_l{i}_ic"] = oracle_prefix_metrics[i].get("return_ic", 0.0)
        metrics[f"oracle_l{i}_icir"] = oracle_prefix_metrics[i].get("return_icir", 0.0)

    if (not is_train) and token_monitor is not None:
        print(f"\n[Token Collapse Check - Epoch {epoch_idx}]")
        token_monitor.compute(verbose=True)
        print("-" * 50)

        oracle_str = " | ".join([f"OracleL{i} RankIC={metrics[f'oracle_l{i}_rankic']:.4f}" for i in range(n_layers)])
        log_str = f"[Epoch {epoch_idx} {mode}] "
        for i, (ce, acc) in enumerate(final_layer_metrics):
            log_str += f"L{i}:[Acc={acc*100:.2f}% CE={ce:.3f}] "
        log_str += f"| OracleFull RankIC={metrics['oracle_rankic']:.4f} | {oracle_str}"
        print(log_str)

    if (not is_train) and collect_signal:
        df_signal = pd.DataFrame(
            {
                "date": all_dates,
                "code": all_codes,
                "signal": np.concatenate(all_signals) if len(all_signals) > 0 else np.array([]),
                "true_mean_price": np.concatenate(all_true_prices) if len(all_true_prices) > 0 else np.array([]),
            }
        )
        return avg_loss, metrics, (avg_ce_loss, avg_trend_loss, final_layer_metrics), avg_recon_error, None, df_signal

    return avg_loss, metrics, (avg_ce_loss, avg_trend_loss, final_layer_metrics), avg_recon_error, None

# ============================================================
# 4) Main
# ============================================================
def main():
    seed = int(CONFIG.get("seed", 42))
    seed_everything(seed)

    CONFIG["stats_input_dim"] = 5

    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["gpt_save_dir"], exist_ok=True)

    vis_dir = os.path.join(CONFIG["gpt_save_dir"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"[*] Using Device: {device}")

    vqvae = VQVAE_Daily(CONFIG).to(device)

    vqvae_path = os.path.join(CONFIG["vqvae_save_dir"], "best_vqvae.pth")
    if not os.path.exists(vqvae_path):
        print(f"[!] Warning: Sorted VQVAE not found at {vqvae_path}. Trying unsorted best_vqvae.pth...")
        vqvae_path = os.path.join(CONFIG["vqvae_save_dir"], "best_vqvae.pth")
        if not os.path.exists(vqvae_path):
            raise FileNotFoundError(f"VQVAE model not found at {vqvae_path}")

    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    print("[*] VQ-VAE loaded.")
    for param in vqvae.parameters():
        param.requires_grad = False

    dfs = [pd.read_csv(p) for p in CONFIG["data_paths"]]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["kdcode", "dt"]).drop_duplicates(subset=["kdcode", "dt"], keep="last").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["dt"])

    split_date = pd.to_datetime(CONFIG["split_date"])

    train_ds = RVQSequenceDataset(
        df[df["dt"] < split_date],
        vqvae,
        CONFIG,
        device=device,
        seq_stride_days=CONFIG.get("seq_stride_days_train", 1),
    )
    val_ds = RVQSequenceDataset(
        df,
        vqvae,
        CONFIG,
        device=device,
        min_decision_date=split_date,
        seq_stride_days=CONFIG.get("seq_stride_days_val", 1),
    )
    print(f"[*] Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["gpt_batch_size"], shuffle=True, num_workers=int(CONFIG.get("num_workers", 4)))
    val_loader = DataLoader(val_ds, batch_size=CONFIG["gpt_batch_size"], shuffle=False, num_workers=int(CONFIG.get("num_workers", 4)))

    model = TimeLLM_Enhanced(CONFIG).to(device)

    # 初始化 TimeLLM 的 embedding 为 VQ-VAE 的 codebook
    try:
        print("[*] Initializing Embeddings from VQ-VAE Codebook...")
        for i in range(model.n_rvq_layers):
            vq_codebook = vqvae.rvq.layers[i]._embedding.weight.data
            if model.source_embeds[i].weight.data.shape != vq_codebook.shape:
                raise RuntimeError(
                    f"Layer {i} codebook shape mismatch: "
                    f"TimeLLM={tuple(model.source_embeds[i].weight.data.shape)} vs "
                    f"VQVAE={tuple(vq_codebook.shape)}"
                )
            model.source_embeds[i].weight.data.copy_(vq_codebook)
            if model.use_market_context:
                model.market_source_embeds[i].weight.data.copy_(vq_codebook)
    except Exception as e:
        print(f"[!] Warning: Embedding initialization failed: {e}")

    # 训练集 token 分布统计
    vocab_sizes_list = parse_n_embeddings_list(CONFIG, model.n_rvq_layers)
    token_dist_samples = int(CONFIG.get("token_dist_samples", 200000))
    token_dist_seed = int(CONFIG.get("token_dist_seed", 123))
    token_dist_topk = int(CONFIG.get("token_dist_topk", 20))
    token_dist_save = CONFIG.get("token_dist_save_json", None)
    if token_dist_save is None:
        token_dist_save = os.path.join(CONFIG["gpt_save_dir"], "train_token_distribution.json")

    _ = compute_train_token_distribution(
        train_ds=train_ds,
        n_layers=model.n_rvq_layers,
        vocab_sizes_list=vocab_sizes_list,
        gpt_context_len=int(CONFIG["gpt_context_len"]),
        samples=token_dist_samples,
        seed=token_dist_seed,
        topk=token_dist_topk,
        save_json_path=token_dist_save,
    )

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "time_gates" in name or param.dim() < 2 or "adaLN_modulation" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"[*] Optimizer Groups: Decay={len(decay_params)} tensors, No-Decay={len(no_decay_params)} tensors")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": CONFIG["gpt_weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG["gpt_lr"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["gpt_epochs"])

    best_rankic = -999.0
    best_epoch_metrics = {}

    for epoch in range(CONFIG["gpt_epochs"]):
        t_loss, _, t_details, t_recon, _ = run_epoch(
            "train", model, vqvae, train_loader, optimizer, device, epoch + 1
        )
        v_loss, v_met, v_details, v_recon, _, val_signal_df = run_epoch(
            "val", model, vqvae, val_loader, None, device, epoch + 1, collect_signal=True, vis_save_dir=vis_dir
        )

        if len(val_signal_df) > 0:
            save_score_dir = os.path.join(CONFIG["gpt_save_dir"], "val_scores", f"epoch_{epoch+1:03d}")
            os.makedirs(save_score_dir, exist_ok=True)
            for dt, group in val_signal_df.groupby("date"):
                if isinstance(dt, (int, np.integer)):
                    dt = pd.to_datetime(dt)
                fname = os.path.join(save_score_dir, f"{dt.strftime('%Y%m%d')}.csv")
                group[["code", "signal"]].to_csv(fname, index=False)

        scheduler.step()

        t_ce_w, t_trnd, t_layer_metrics = t_details
        v_ce_w, v_trnd, v_layer_metrics = v_details

        # 打印逐层 Oracle RankIC（前缀）
        oracle_layer_str = " ".join(
            [f"OracleL{i}:{v_met.get(f'oracle_l{i}_rankic', 0.0):.4f}" for i in range(model.n_rvq_layers)]
        )

        print(
            f"Ep {epoch+1:02d} | "
            f"T_Loss:{t_loss:.3f} | V_Loss:{v_loss:.3f} | "
            f"IC:{v_met.get('return_ic',0):.4f} RankIC:{v_met.get('return_rankic',0):.4f} "
            f"ICIR:{v_met.get('return_icir',0):.2f} | "
            f"OracleFull:{v_met.get('oracle_rankic',0):.4f} | {oracle_layer_str} | "
            f"T_Recon:{t_recon:.4f} V_Recon:{v_recon:.4f}"
        )

        train_log = "Train: "
        for i, (ce, acc) in enumerate(t_layer_metrics):
            train_log += f"[L{i} Acc:{acc*100:.1f}%] "
        val_log = "Val  : "
        for i, (ce, acc) in enumerate(v_layer_metrics):
            val_log += f"[L{i} Acc:{acc*100:.1f}%] "
        print(f"         | {train_log}")
        print(f"         | {val_log}")

        current_rankic = v_met.get("return_rankic", -1.0)
        if current_rankic > best_rankic:
            best_rankic = current_rankic
            torch.save(model.state_dict(), os.path.join(CONFIG["gpt_save_dir"], "best_gen_model.pth"))
            print(f"    >>> Saved Best Model (RankIC: {best_rankic:.4f})")

            best_epoch_metrics = {
                "epoch": epoch + 1,
                "rank_ic": best_rankic,
                "ic": v_met.get("return_ic", 0.0),
                "icir": v_met.get("return_icir", 0.0),
                "val_loss": v_loss,
                "val_recon_mse": v_recon,
                "oracle_full_rankic": v_met.get("oracle_rankic", 0.0),
            }
            for i, (ce, acc) in enumerate(v_layer_metrics):
                best_epoch_metrics[f"val_l{i}_acc"] = acc
                best_epoch_metrics[f"val_l{i}_ce"] = ce
                best_epoch_metrics[f"oracle_l{i}_rankic"] = v_met.get(f"oracle_l{i}_rankic", 0.0)

    print("\n" + "=" * 50)
    print(" [*] Training Completed. Best Model Report:")
    if best_epoch_metrics:
        print(f"    - Best Epoch         : {best_epoch_metrics['epoch']}")
        print(f"    - Best RankIC        : {best_epoch_metrics['rank_ic']:.4f}")
        print(f"    - Best IC        : {best_epoch_metrics['ic']:.4f}")
        print(f"    - Best ICIR          : {best_epoch_metrics['icir']:.2f}")
        print(f"    - OracleFull RankIC  : {best_epoch_metrics.get('oracle_full_rankic', 0.0):.4f}")
        for i in range(int(CONFIG.get("rvq_layers", 3))):
            k = f"oracle_l{i}_rankic"
            if k in best_epoch_metrics:
                print(f"    - {k}       : {best_epoch_metrics[k]:.4f}")
    else:
        print("    - No metrics recorded.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()