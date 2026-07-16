import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# 导入配置和模型定义
from config import CONFIG
from model_rqvae import VQVAE_Daily
from model_gen import TimeLLM_Enhanced

# 假设 train.py 在同一目录下，且包含 RVQSequenceDataset 类
# 如果不在，请将 RVQSequenceDataset 类的代码复制到这里
from train import RVQSequenceDataset

# ============================================================
# 1. 引入指标计算类 (从训练代码复制，确保逻辑一致)
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

        # 按日期分组计算 IC / RankIC
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
        
        return {
            "return_ic": mean_ic, 
            "return_rankic": mean_rank_ic, 
            "return_icir": icir,
            "daily_metrics": daily_metrics # 可选：返回每日明细
        }

# ============================================================
# 2. 模型加载函数
# ============================================================
def load_models(device):
    print(f"[*] Loading models on {device}...")
    
    # 1. Load VQ-VAE
    vqvae = VQVAE_Daily(CONFIG).to(device)
    vqvae_path = os.path.join(CONFIG["vqvae_save_dir"], "best_vqvae_sorted.pth")
    if not os.path.exists(vqvae_path):
        vqvae_path = os.path.join(CONFIG["vqvae_save_dir"], "best_vqvae.pth")
    
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        print(f"    [-] VQ-VAE loaded from {vqvae_path}")
    else:
        raise FileNotFoundError(f"VQ-VAE model not found at {vqvae_path}")
    vqvae.eval()

    # 2. Load TimeLLM
    model = TimeLLM_Enhanced(CONFIG).to(device)
    gen_model_path = os.path.join(CONFIG["gpt_save_dir"], "best_gen_model.pth")
    
    if os.path.exists(gen_model_path):
        model.load_state_dict(torch.load(gen_model_path, map_location=device))
        print(f"    [-] TimeLLM loaded from {gen_model_path}")
    else:
        raise FileNotFoundError(f"Generative model not found at {gen_model_path}")
    model.eval()
    
    return vqvae, model

# ============================================================
# 3. 推理主流程 (含指标计算)
# ============================================================
def run_inference_with_metrics(start_date=None, end_date=None, output_dir="./inference_scores"):
    device = torch.device(CONFIG["device"])
    CONFIG["stats_input_dim"] = 5
    
    vqvae, model = load_models(device)
    
    # 加载数据
    print("[*] Loading data for inference...")
    dfs = [pd.read_csv(p) for p in CONFIG["data_paths"]]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["kdcode", "dt"]).drop_duplicates(subset=["kdcode", "dt"], keep="last").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["dt"])

    predict_start_dt = pd.to_datetime(start_date) if start_date else pd.to_datetime(CONFIG["split_date"])
    
    # 初始化 Dataset
    # 注意：为了计算 IC，Dataset 必须包含未来的数据作为 label
    # 因此这通常用于回测（Backtest）阶段，而非纯粹的未来预测（Live）
    test_ds = RVQSequenceDataset(
        df,
        vqvae,
        CONFIG,
        device=device,
        min_decision_date=predict_start_dt,
        seq_stride_days=1 
    )
    
    if len(test_ds) == 0:
        print("[!] No samples found.")
        return

    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG["gpt_batch_size"], 
        shuffle=False, 
        num_workers=int(CONFIG.get("num_workers", 4))
    )

    print(f"[*] Starting inference on {len(test_ds)} samples...")
    
    # 初始化记录器
    metric_recorder = KronosMetricRecorder()
    
    all_signals = []
    all_dates = []
    all_codes = []
    all_trues = [] # 保存真实值以便后续分析（可选）

    n_layers = model.n_rvq_layers
    gpt_context_len = int(CONFIG["gpt_context_len"])
    H_pred = int(CONFIG.get("pred_horizon", 4))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # 解包包含 true_close_patch (真实未来价格)
            (tokens, mu, sd, date_ns, true_close_patch, last_close, kdcode, time_feats, market_tokens) = batch
            
            tokens = tokens.to(device)
            mu = mu.to(device)
            sd = sd.to(device)
            last_close = last_close.to(device)
            time_feats = time_feats.to(device)
            market_tokens = market_tokens.to(device)
            true_close_patch = true_close_patch.to(device) # 需要放到 GPU 上参与计算

            if tokens.shape[1] <= gpt_context_len:
                continue

            # 准备输入
            context_tokens = tokens[:, :gpt_context_len, :]
            curr_market = market_tokens[:, :gpt_context_len, :] if market_tokens is not None else None
            curr_time = time_feats[:, :gpt_context_len, :] if time_feats is not None else None
            
            # 模型预测
            logits_inf_list = model(
                context_tokens,
                market_ids=curr_market,
                time_feats=curr_time,
                target_ids=None,
                stats_feats=sd,
            )

            last_step_logits_list = [l[:, -1, :] for l in logits_inf_list]

            # Soft Reconstruction 计算
            mu_c = mu[:, 3].unsqueeze(1)
            sd_c = sd[:, 3].unsqueeze(1)

            z_soft_sum = 0.0
            for i in range(n_layers):
                probs = F.softmax(last_step_logits_list[i], dim=-1)
                codebook_w = vqvae.rvq.layers[i]._embedding.weight
                z_soft_sum = z_soft_sum + torch.matmul(probs, codebook_w)

            z_soft = z_soft_sum.unsqueeze(-1)
            recon_norm = vqvae.decoder(z_soft)
            
            pred_close_norm = recon_norm[:, 3, :H_pred]
            pred_price = pred_close_norm * sd_c + mu_c

            # === 计算信号 (预测收益率) ===
            pred_mean_signal = (pred_price.mean(dim=1) - last_close) / (last_close + 1e-8)
            
            # === 计算真实标签 (真实收益率) ===
            # 这里取未来 H_pred 天的均价相对于昨收的收益率，与训练逻辑一致
            true_mean_label = (true_close_patch[:, :H_pred].mean(dim=1) - last_close) / (last_close + 1e-8)

            # === 更新指标记录器 ===
            metric_recorder.update(pred_mean_signal, true_mean_label, date_ns)

            # 收集结果用于保存文件
            all_signals.append(pred_mean_signal.cpu().numpy())
            all_dates.extend(date_ns.cpu().numpy())
            all_codes.extend(kdcode)
            all_trues.append(true_mean_label.cpu().numpy())

    # 4. 计算并打印指标
    print("\n" + "="*50)
    print("[*] Computing Metrics...")
    metrics = metric_recorder.compute()
    
    print(f"    RankIC : {metrics['return_rankic']:.4f}")
    print(f"    IC     : {metrics['return_ic']:.4f}")
    print(f"    ICIR   : {metrics['return_icir']:.4f}")
    print("="*50 + "\n")

    # 5. 保存结果文件 (按日期保存 CSV)
    print("[*] Saving results...")
    
    if not all_signals:
        return

    full_signal = np.concatenate(all_signals)
    full_true = np.concatenate(all_trues)
    
    df_result = pd.DataFrame({
        "date": all_dates,
        "code": all_codes,
        "signal": full_signal,
        "true_ret": full_true # 可选：把真实收益也保存下来，方便后续复查
    })
    
    df_result["date"] = pd.to_datetime(df_result["date"])
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df_result = df_result[df_result["date"] <= end_dt]

    os.makedirs(output_dir, exist_ok=True)
    grouped = df_result.groupby("date")
    
    for dt, group in tqdm(grouped, desc="Writing CSVs"):
        date_str = dt.strftime('%Y%m%d')
        fname = os.path.join(output_dir, f"{date_str}.csv")
        
        # 保存列：保留 code 和 signal，额外加上 true_ret 以便分析
        save_cols = ["code", "signal", "true_ret"]
        save_df = group[save_cols].sort_values("code")
        
        save_df.to_csv(fname, index=False)
        
    print(f"[*] Done. Metrics computed and files saved to {output_dir}")

if __name__ == "__main__":
    # 配置
    OUTPUT_DIR = CONFIG["gpt_save_dir"] + "/prediction"
    START_DATE = "2025-01-01" 
    END_DATE = None 

    run_inference_with_metrics(
        start_date=START_DATE, 
        end_date=END_DATE, 
        output_dir=OUTPUT_DIR
    )