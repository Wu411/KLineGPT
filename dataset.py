import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import CONFIG
from tqdm import tqdm

class DailyPatchDataset(Dataset):
    """
    Tokenizer 训练用 Dataset

    归一化策略（必须与下游一致）：
    - 每只股票、每个通道独立 rolling mean/std，并 shift(1) 保证因果
    - 对于 patch 起点 start，patch 内所有点使用同一对 mu[start], std[start]
    - volume: log1p 后再 rolling z-score
    """

    def __init__(self, df, mode="train"):
        self.window_size = int(CONFIG["window_size"])
        self.stride = int(CONFIG["stride"]) if mode == "train" else int(CONFIG["window_size"])
        self.mode = mode

        self.norm_window = int(CONFIG.get("norm_window", 60))
        self.clip_value = float(CONFIG.get("norm_clip", 10.0))
        self.eps = float(CONFIG.get("norm_eps", 1e-6))

        # 建议这里保证有序
        self.df = df.copy()
        self.df["dt"] = pd.to_datetime(self.df["dt"])
        self.df = self.df.sort_values(["kdcode", "dt"]).reset_index(drop=True)

        # 原始通道
        self.o = self.df["open"].to_numpy(np.float32)
        self.h = self.df["high"].to_numpy(np.float32)
        self.l = self.df["low"].to_numpy(np.float32)
        self.c = self.df["close"].to_numpy(np.float32)
        self.v = self.df["volume"].to_numpy(np.float32)

        # patch 起点索引（全局 index）
        self.indices = []

        stock_groups = self.df.groupby("kdcode").indices
        print(f"[*] Processing {len(stock_groups)} stocks for {mode} (rolling norm)...")

        valid = []
        for kdcode, idx_array in stock_groups.items():
            idx_array = np.asarray(idx_array, dtype=np.int64)
            stock_len = len(idx_array)

            # rolling stats shift(1) => 起点至少 norm_window
            min_s_rel = self.norm_window
            max_s_rel = stock_len - self.window_size
            if max_s_rel < min_s_rel:
                continue

            starts_rel = np.arange(min_s_rel, max_s_rel + 1, self.stride, dtype=np.int64)
            if len(starts_rel) == 0:
                continue

            # 稳健映射：不要假设 idx_array 连续
            starts_abs = idx_array[starts_rel]
            valid.append(starts_abs)

        if len(valid) > 0:
            self.indices = np.concatenate(valid).astype(np.int64)

        print(f"[*] {mode} dataset size: {len(self.indices)}")

        # 预计算每个全局 index 的 rolling mu/std（5 通道）
        self._mu = np.zeros((len(self.df), 5), dtype=np.float32)
        self._sd = np.ones((len(self.df), 5), dtype=np.float32)

        for kdcode, idx_array in stock_groups.items():
            idx_array = np.asarray(idx_array, dtype=np.int64)

            def roll_mu_sd(x: np.ndarray):
                s = pd.Series(x, copy=False)
                mu = s.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32)
                sd = s.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                return mu, sd

            o = self.o[idx_array]
            h = self.h[idx_array]
            l = self.l[idx_array]
            c = self.c[idx_array]
            v = np.log1p(np.maximum(self.v[idx_array], 0.0)).astype(np.float32)

            o_mu, o_sd = roll_mu_sd(o)
            h_mu, h_sd = roll_mu_sd(h)
            l_mu, l_sd = roll_mu_sd(l)
            c_mu, c_sd = roll_mu_sd(c)
            v_mu, v_sd = roll_mu_sd(v)

            self._mu[idx_array, 0] = o_mu
            self._mu[idx_array, 1] = h_mu
            self._mu[idx_array, 2] = l_mu
            self._mu[idx_array, 3] = c_mu
            self._mu[idx_array, 4] = v_mu

            self._sd[idx_array, 0] = o_sd
            self._sd[idx_array, 1] = h_sd
            self._sd[idx_array, 2] = l_sd
            self._sd[idx_array, 3] = c_sd
            self._sd[idx_array, 4] = v_sd

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = int(self.indices[idx])
        end = start + self.window_size

        o_patch = self.o[start:end]
        h_patch = self.h[start:end]
        l_patch = self.l[start:end]
        c_patch = self.c[start:end]
        v_patch = np.log1p(np.maximum(self.v[start:end], 0.0)).astype(np.float32)

        mu = self._mu[start].astype(np.float32)  # [5]
        sd = self._sd[start].astype(np.float32)  # [5]

        # 防御：sd 太小或非有限
        sd = np.where(np.isfinite(sd) & (sd > self.eps), sd, 1.0).astype(np.float32)
        mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)

        norm_o = (o_patch - mu[0]) / sd[0]
        norm_h = (h_patch - mu[1]) / sd[1]
        norm_l = (l_patch - mu[2]) / sd[2]
        norm_c = (c_patch - mu[3]) / sd[3]
        norm_v = (v_patch - mu[4]) / sd[4]

        patch = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=1).astype(np.float32)
        patch = np.clip(patch, -self.clip_value, self.clip_value)

        if np.isnan(patch).any() or np.isinf(patch).any():
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

        # 返回 [C, T]
        return torch.tensor(patch.T, dtype=torch.float32)



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
        self.indices = []; self.stock_tokens = []; self.stock_close_mu = []; self.stock_close_sd = []; self.stock_true_next_ohlc = []; self.stock_kdcode = []

        vqvae_model.eval().to(device)
        df = df.copy().sort_values(["kdcode", "dt"]).reset_index(drop=True)
        groups = df.groupby("kdcode").indices
        o_all = df["open"].to_numpy(np.float32); h_all = df["high"].to_numpy(np.float32); l_all = df["low"].to_numpy(np.float32)
        c_all = df["close"].to_numpy(np.float32); v_all = df["volume"].to_numpy(np.float32); dt_all = df["dt"].to_numpy("datetime64[ns]")
        
        self.latent_len = None 
        print("[*] Encoding stocks & Preparing Data...")
        with torch.no_grad():
            for kdcode, idx_array in tqdm(groups.items()):
                idx_array = np.asarray(idx_array, dtype=np.int64)
                n = len(idx_array)
                if n <= (self.norm_window + self.window_size + 6): continue
                o_series = pd.Series(o_all[idx_array]); h_series = pd.Series(h_all[idx_array])
                l_series = pd.Series(l_all[idx_array]); c_series = pd.Series(c_all[idx_array])
                v_series = pd.Series(np.log1p(np.maximum(v_all[idx_array], 0.0)).astype(np.float32))
                o_mu = o_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32); o_sd = o_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                h_mu = h_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32); h_sd = h_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                l_mu = l_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32); l_sd = l_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                c_mu = c_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32); c_sd = c_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                v_mu = v_series.rolling(self.norm_window).mean().shift(1).to_numpy(np.float32); v_sd = v_series.rolling(self.norm_window).std(ddof=0).shift(1).to_numpy(np.float32)
                valid_mask = (np.isfinite(o_mu) & (o_sd > 1e-8) & np.isfinite(h_mu) & (h_sd > 1e-8) & np.isfinite(l_mu) & (l_sd > 1e-8) & np.isfinite(c_mu) & (c_sd > 1e-8) & np.isfinite(v_mu) & (v_sd > 1e-8))
                valid_starts = np.where(valid_mask)[0]
                valid_starts = valid_starts[(valid_starts >= self.norm_window) & (valid_starts <= n - self.window_size - 5)]
                if len(valid_starts) == 0: continue
                stock_codes_list = []
                for b0 in range(0, len(valid_starts), self.encode_batch_size):
                    batch = []
                    for s in valid_starts[b0:b0+self.encode_batch_size]:
                        e = s + self.window_size
                        mu = np.array([o_mu[s], h_mu[s], l_mu[s], c_mu[s], v_mu[s]], dtype=np.float32)
                        sd = np.array([max(x, self.eps) for x in [o_sd[s], h_sd[s], l_sd[s], c_sd[s], v_sd[s]]], dtype=np.float32)
                        o_patch = o_all[idx_array[s:e]]; h_patch = h_all[idx_array[s:e]]; l_patch = l_all[idx_array[s:e]]; c_patch = c_all[idx_array[s:e]]; v_patch = np.log1p(np.maximum(v_all[idx_array[s:e]], 0.0)).astype(np.float32)
                        norm_o = (o_patch - mu[0]) / sd[0]; norm_h = (h_patch - mu[1]) / sd[1]; norm_l = (l_patch - mu[2]) / sd[2]; norm_c = (c_patch - mu[3]) / sd[3]; norm_v = (v_patch - mu[4]) / sd[4]
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
                self.stock_close_mu.append(mu_arr[valid_starts]); self.stock_close_sd.append(sd_arr[valid_starts])
                current_ohlc_chunk = []; 
                for s in valid_starts: current_ohlc_chunk.append(c_all[idx_array[s : s + self.window_size]])
                self.stock_true_next_ohlc.append(current_ohlc_chunk); self.stock_kdcode.append(str(kdcode))
                n_patches = len(valid_starts)
                max_start_idx = n_patches - 1 - (self.num_patches_total - 1) * self.window_size
                if max_start_idx < 0: continue
                for p0 in range(0, max_start_idx + 1, self.seq_stride_days):
                    target_relative_idx = (self.num_patches_total - 1) * self.window_size
                    target_patch_start_idx_in_valid = p0 + target_relative_idx
                    global_idx_start = valid_starts[target_patch_start_idx_in_valid]
                    decision_dt = dt_all[idx_array[global_idx_start]]
                    if self.min_decision_date and decision_dt < self.min_decision_date: continue
                    self.indices.append((len(self.stock_tokens)-1, p0, decision_dt.astype(np.int64)))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        seq_idx, p0, date_ns = self.indices[i]
        patch_indices = [p0 + k * self.window_size for k in range(self.num_patches_total)]
        tokens = self.stock_tokens[seq_idx][patch_indices]; tokens = tokens.reshape(-1, tokens.shape[-1])
        target_valid_start_idx = patch_indices[-1]
        mu = self.stock_close_mu[seq_idx][target_valid_start_idx]; sd = self.stock_close_sd[seq_idx][target_valid_start_idx]
        true_close_patch = self.stock_true_next_ohlc[seq_idx][target_valid_start_idx]; kdcode = self.stock_kdcode[seq_idx]
        return (torch.tensor(tokens, dtype=torch.long), torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32),
            torch.tensor(date_ns, dtype=torch.long), torch.tensor(true_close_patch, dtype=torch.float32), kdcode)