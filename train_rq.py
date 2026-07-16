import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# 假设 dataset.py 和 model_rqvae.py 已经更新
from config import CONFIG
from model_rqvae import VQVAE_Daily
from dataset import DailyPatchDataset


# ============================================================
# [修改功能] Codebook 拓扑排序与可视化
# ============================================================
def sort_codebook_by_pca(model):
    """
    使用 PCA 对模型中的 Codebook 进行重排序。
    [深度修正]：合并遍历逻辑，确保 EMA 统计量与 Embedding 权重严格同步重排。
    """
    print("\n[*] Sorting codebook indices via PCA (Topological Reordering)...")
    sorted_layers = 0
    
    with torch.no_grad():
        for name, module in model.named_modules():
            # 仅拦截 VectorQuantizerEMA 模块，避免被底层的 nn.Embedding 重复触发
            if "VectorQuantizerEMA" in module.__class__.__name__:
                # 提取底层权重
                weights = module._embedding.weight.data
                N, D = weights.shape
                if N <= 1: continue
                
                # 1. 计算 PCA 第一主成分
                mean = weights.mean(dim=0, keepdim=True)
                centered = weights - mean
                
                try:
                    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                    pc1 = Vh[0, :]
                except RuntimeError as e:
                    print(f"    [!] PCA failed for {name}: {e}")
                    continue

                # 2. 投影到一维标量并获取排序索引
                projections = (centered @ pc1.unsqueeze(-1)).squeeze(-1)
                sort_idx = torch.argsort(projections)
                
                # 3. [关键同步]：同时重排 Embedding 和所有 EMA 追踪器
                module._embedding.weight.data = weights[sort_idx]
                
                if hasattr(module, '_ema_w'):
                    module._ema_w.data = module._ema_w.data[sort_idx]
                if hasattr(module, '_ema_cluster_size'):
                    module._ema_cluster_size.data = module._ema_cluster_size.data[sort_idx]
                
                print(f"    [-] Layer '{name}' sorted. (Vocab={N}, Dim={D})")
                sorted_layers += 1
                
    if sorted_layers == 0:
        print("    [!] Warning: No Codebook layers found or sorted.")
    else:
        print(f"    [*] Successfully sorted {sorted_layers} codebooks.")


def plot_codebook_similarity(model, save_path, title="Codebook Cosine Similarity"):
    model.eval()
    embeddings = []
    names = []
    
    with torch.no_grad():
        for name, module in model.named_modules():
            # 过滤掉一些无关的 embedding（如果有的话）
            if isinstance(module, torch.nn.Embedding) and module.num_embeddings > 1:
                embeddings.append(module.weight.data.cpu())
                names.append(name)
    
    if not embeddings:
        return

    n_layers = len(embeddings)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5))
    if n_layers == 1: axes = [axes]
    
    for i, weight in enumerate(embeddings):
        # 计算 Cosine Similarity
        norm_w = F.normalize(weight, p=2, dim=1)
        sim_matrix = torch.mm(norm_w, norm_w.t()).numpy()
        
        # 如果是 ax 数组
        ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
        
        im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"L{i} (Size {weight.shape[0]})")
        ax.axis('off') # 关掉坐标轴刻度，太密了看不清
        
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [-] Similarity matrix saved to {save_path}")


# ============================================================
# 数据处理与 Metric 逻辑 (适配 List)
# ============================================================

def prepare_data(config):
    paths = config.get("tokenizer_data_path", [])
    
    if not paths:
        raise ValueError("No data paths provided in config.")

    print(f"[*] Loading data from {len(paths)} files...")
    
    dfs = []
    for p in paths:
        if os.path.exists(p):
            df_temp = pd.read_csv(p)
            dfs.append(df_temp)
        else:
            print(f"[!] Warning: File path {p} does not exist, skipping.")
    
    if not dfs:
        raise ValueError("No valid data files loaded.")

    full_df = pd.concat(dfs, ignore_index=True)
    
    print("[*] Processing datetime and sorting...")
    full_df["dt"] = pd.to_datetime(full_df["dt"])
    full_df = full_df.sort_values(["kdcode", "dt"]).reset_index(drop=True)

    split_date = pd.to_datetime(config["split_date"])
    train_df = full_df[full_df["dt"] < split_date].copy()
    val_df = full_df[full_df["dt"] >= split_date].copy()
    
    print(f"[*] Data loaded. Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def calculate_codebook_metrics(codes, num_embeddings_list):
    """
    计算 Codebook 利用率和困惑度 (Perplexity)。
    codes: [B, T, D]
    num_embeddings_list: List[int], 每层的码本大小
    """
    if codes.ndim == 2:
        codes = codes.unsqueeze(-1)
        
    B, T, D = codes.shape
    metrics = {}
    
    total_active_ratio = 0
    total_ppl = 0
    
    for d in range(D):
        # [修改] 获取当前层的正确码本大小
        n_emb = num_embeddings_list[d]
        
        flat_codes = codes[..., d].reshape(-1).float()
        
        unique_codes = torch.unique(flat_codes)
        n_active = unique_codes.numel()
        
        # 使用当前层的 n_emb 作为 bins 数量
        counts = torch.histc(flat_codes, bins=n_emb, min=0, max=n_emb-1)
        probs = counts / (counts.sum() + 1e-9)
        probs = probs[probs > 0] # 只计算非零概率
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        perplexity = torch.exp(entropy).item()
        
        metrics[f"Layer_{d}/Active"] = n_active
        metrics[f"Layer_{d}/Usage"] = n_active / n_emb # 记录使用率而非绝对值，方便跨层比较
        metrics[f"Layer_{d}/PPL"] = perplexity
        
        total_active_ratio += (n_active / n_emb)
        total_ppl += perplexity

    metrics["Avg/Usage"] = total_active_ratio / D
    metrics["Avg/PPL"] = total_ppl / D
    
    return metrics


def check_collapse(metrics, num_embeddings_list, threshold_ratio=0.05):
    """
    检查是否有层的 Codebook 使用率低于阈值。
    """
    collapse_warnings = []
    
    # 遍历所有记录了 Active 数量的 key
    for d, n_emb in enumerate(num_embeddings_list):
        key = f"Layer_{d}/Active"
        if key in metrics:
            active_count = metrics[key]
            threshold = n_emb * threshold_ratio
            
            if active_count < threshold:
                collapse_warnings.append(f"L{d}({int(active_count)}/{n_emb})")
            
    return collapse_warnings


def log_token_histograms(writer, codes, step, num_embeddings_list):
    if codes.ndim == 2:
        codes = codes.unsqueeze(-1)
    
    B, T, D = codes.shape
    for d in range(D):
        n_emb = num_embeddings_list[d]
        layer_codes = codes[..., d].reshape(-1).float()
        
        # Tensorboard histogram
        writer.add_histogram(f"Token_Dist/Layer_{d}", layer_codes, step, bins=n_emb)


@torch.no_grad()
def visualize_rvq(model, dataset, device, num_samples=4, save_path=None, title_prefix="Val"):
    # 可视化逻辑通常不需要改动，因为它只关心重构出的值
    model.eval()
    if len(dataset) == 0:
        return None

    actual_samples = min(len(dataset), num_samples)
    indices = np.random.choice(len(dataset), actual_samples, replace=False)
    
    batch_tensors = []
    meta_infos = [] 
    
    for idx in indices:
        # 兼容 Dataset 实现差异，假设 dataset 可以直接通过 idx 获取 tensor 和 meta
        # 这里沿用你原始逻辑
        item = dataset[idx] # tensor
        batch_tensors.append(item)
        
        # 假设 dataset 有内部属性 _mu, _sd
        # 注意：如果 Dataset 做了 shuffle，这里的 idx 可能对应不上原始 df 的 idx
        # 但通常 Dataset[i] 是确定的。这里假设 dataset.indices[i] 映射回原始数据
        if hasattr(dataset, 'indices'):
            start = int(dataset.indices[idx])
            mu = dataset._mu[start].astype(np.float32)
            sd = dataset._sd[start].astype(np.float32)
        else:
            # Fallback mock
            mu = np.zeros(5, dtype=np.float32)
            sd = np.ones(5, dtype=np.float32)
            
        eps = float(CONFIG.get("norm_eps", 1e-6))
        sd = np.where(np.isfinite(sd) & (sd > eps), sd, 1.0).astype(np.float32)
        mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
        
        meta_infos.append((mu, sd))

    x = torch.stack(batch_tensors).to(device)
    loss_vq, x_recon, _ = model(x)

    x_np = x.detach().cpu().numpy()
    xr_np = x_recon.detach().cpu().numpy()
    
    W = x_np.shape[2]
    fig, axes = plt.subplots(3, actual_samples, figsize=(4 * actual_samples, 10), sharex=True, squeeze=False)
    
    for i in range(actual_samples):
        curr_x = x_np[i]
        curr_xr = xr_np[i]
        curr_mu, curr_sd = meta_infos[i]
        
        # Denormalize
        x_denorm = (curr_x * curr_sd[:, None]) + curr_mu[:, None]
        xr_denorm = (curr_xr * curr_sd[:, None]) + curr_mu[:, None]

        norm_real = curr_x[3]
        norm_recon = curr_xr[3]
        price_real = x_denorm[3]
        price_recon = xr_denorm[3]
        vol_real = np.expm1(np.clip(x_denorm[4], -50, 50))
        vol_recon = np.expm1(np.clip(xr_denorm[4], -50, 50))
        
        ax0 = axes[0, i]
        ax0.plot(norm_real, label="True", linewidth=1.5, alpha=0.7, color='black')
        ax0.plot(norm_recon, label="Recon", linewidth=1.5, linestyle="--", color='red')
        ax0.set_title(f"Spl {indices[i]}: Norm Close")
        ax0.grid(True, alpha=0.3)
        if i == 0: ax0.legend(fontsize='small') 

        ax1 = axes[1, i]
        ax1.plot(price_real, label="True", linewidth=1.5, alpha=0.7, color='blue')
        ax1.plot(price_recon, label="Recon", linewidth=1.5, linestyle="--", color='orange')
        ax1.set_title("Real Price")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[2, i]
        ax2.bar(np.arange(W), vol_real, alpha=0.3, color='gray', label="True")
        ax2.plot(vol_recon, label="Recon", linewidth=1.5, linestyle="--", color='green')
        ax2.set_title("Volume")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    return fig


def train_epoch(model, dataloader, optimizer, device, num_embeddings_list):
    model.train()
    stats = {"loss": 0.0, "recon": 0.0, "vq": 0.0}
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0], device=device).view(1, 5, 1)

    for x in tqdm(dataloader, desc="Train", leave=False):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)

        loss_vq, x_recon, codes = model(x)
        loss_recon = torch.mean(weights * (x_recon - x) ** 2)
        loss = loss_recon + loss_vq

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # [修改] 传入 list
        batch_metrics = calculate_codebook_metrics(codes, num_embeddings_list)

        stats["loss"] += loss.item()
        stats["recon"] += loss_recon.item()
        stats["vq"] += loss_vq.item()
        
        for k, v in batch_metrics.items():
            if k not in stats:
                stats[k] = 0.0
            stats[k] += v

    n = max(len(dataloader), 1)
    for k in stats:
        stats[k] /= n
    return stats


@torch.no_grad()
def validate_epoch(model, dataloader, device, num_embeddings_list, writer=None, epoch=0):
    model.eval()
    stats = {
        "loss": 0.0, "recon": 0.0, "vq": 0.0, 
        "mse_close": 0.0, "mse_vol": 0.0
    }
    weights = torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0], device=device).view(1, 5, 1)
    last_codes = None 

    for x in tqdm(dataloader, desc="Val", leave=False):
        x = x.to(device)
        loss_vq, x_recon, codes = model(x)
        last_codes = codes 
        
        loss_recon = torch.mean(weights * (x_recon - x) ** 2)
        loss = loss_recon + loss_vq
        
        mse_close = F.mse_loss(x_recon[:, 3, :], x[:, 3, :])
        mse_vol = F.mse_loss(x_recon[:, 4, :], x[:, 4, :])

        # [修改] 传入 list
        batch_metrics = calculate_codebook_metrics(codes, num_embeddings_list)

        stats["loss"] += loss.item()
        stats["recon"] += loss_recon.item()
        stats["vq"] += loss_vq.item()
        stats["mse_close"] += mse_close.item()
        stats["mse_vol"] += mse_vol.item()
        
        for k, v in batch_metrics.items():
            if k not in stats:
                stats[k] = 0.0
            stats[k] += v

    n = max(len(dataloader), 1)
    for k in stats:
        stats[k] /= n
        
    if writer is not None and last_codes is not None:
        log_token_histograms(writer, last_codes, epoch, num_embeddings_list)
        
    return stats


def main():
    if "norm_window" not in CONFIG:
        raise ValueError("CONFIG must include norm_window for rolling normalization.")

    save_dir = CONFIG.get("vqvae_save_dir", "./rqvae_ohlcv")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    device = torch.device(CONFIG.get("device", "cuda:0"))

    train_df, val_df = prepare_data(CONFIG)

    train_ds = DailyPatchDataset(train_df, mode="train")
    val_ds = DailyPatchDataset(val_df, mode="val")

    num_workers = int(CONFIG.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=int(CONFIG.get("vqvae_batch_size", 256)),
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(CONFIG.get("vqvae_batch_size", 256)),
        shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    # 1. 初始化模型
    model = VQVAE_Daily(CONFIG).to(device)
    
    # 2. [关键修改] 动态获取每一层的码本大小
    # 假设 model.rvq.layers 是 nn.ModuleList[VectorQuantizerEMA]
    # 我们直接从模型实例中读取，这样最稳健，无论 config 里写的是 list 还是 int
    if hasattr(model, "rvq") and hasattr(model.rvq, "layers"):
        codebook_sizes = [layer._num_embeddings for layer in model.rvq.layers]
    else:
        # Fallback (为了防止模型结构命名不一致)
        print("[!] Warning: Could not detect variable codebook sizes from model structure. Using config fallback.")
        default_size = int(CONFIG.get("n_embeddings", 256))
        n_layers = int(CONFIG.get("rvq_layers", 4))
        codebook_sizes = [default_size] * n_layers
    
    print(f"[*] Detected Codebook Sizes per Layer: {codebook_sizes}")

    optimizer = optim.AdamW(model.parameters(), lr=float(CONFIG.get("vqvae_lr", 3e-4)), weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    epochs = int(CONFIG.get("vqvae_epochs", 50))
    save_every = int(CONFIG.get("save_every", 5))
    num_viz_samples = int(CONFIG.get("num_viz_samples", 4)) 

    # 早停参数
    patience = int(CONFIG.get("early_stop_patience", 10))
    min_delta = float(CONFIG.get("early_stop_min_delta", 1e-4))
    patience_counter = 0
    best_loss = float("inf")

    print(f"[*] Training RQ-VAE on {device}. Win={CONFIG['window_size']}")
    print(f"[*] Early Stopping: Patience={patience}, Min Delta={min_delta}")

    # ================= 训练循环 =================
    for epoch in range(epochs):
        # [修改] 传入 codebook_sizes 列表
        trn = train_epoch(model, train_loader, optimizer, device, codebook_sizes)
        val = validate_epoch(model, val_loader, device, codebook_sizes, writer, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']

        # TensorBoard Logging
        writer.add_scalar("Loss/Total/Train", trn["loss"], epoch)
        writer.add_scalar("Loss/Total/Val", val["loss"], epoch)
        writer.add_scalar("Loss/Recon/Val", val["recon"], epoch)
        writer.add_scalar("Metrics/MSE_Close", val["mse_close"], epoch)
        writer.add_scalar("Metrics/MSE_Vol", val["mse_vol"], epoch)
        writer.add_scalar("Hyper/LR", current_lr, epoch)
        
        # 记录每层的 PPL 和 Usage
        for k, v in val.items():
            if "Layer" in k or "Avg" in k:
                writer.add_scalar(f"Codebook_Val/{k}", v, epoch)

        # 构造日志字符串
        layer_keys = [k for k in val.keys() if "Layer" in k and "PPL" in k]
        layer_keys.sort()
        ppl_str = ""
        if layer_keys:
            # 简化显示 PPL
            ppls = [f"L{i}:{val[k]:.1f}" for i, k in enumerate(layer_keys)]
            ppl_str = " | PPL " + " ".join(ppls)

        # [修改] 使用新的 check_collapse 逻辑
        collapse_warnings = check_collapse(val, codebook_sizes, threshold_ratio=0.03)
        status_str = "OK"
        if collapse_warnings:
            status_str = f"!! WARN: {', '.join(collapse_warnings)} COLLAPSE !!"

        print(
            f"Ep {epoch+1:03d} | "
            f"Loss {val['loss']:.4f} | "
            f"Status: {status_str}"
            f"{ppl_str} | LR {current_lr:.1e}"
        )

        scheduler.step(val["loss"])

        # 早停与保存
        if val["loss"] < (best_loss - min_delta):
            best_loss = val["loss"]
            patience_counter = 0 
            
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vqvae.pth"))
            print("    [-] New best model saved.")

            fig = visualize_rvq(
                model, val_ds, device,
                num_samples=num_viz_samples,
                save_path=os.path.join(save_dir, f"viz_best_ep{epoch+1}.png"),
                title_prefix=f"Best Ep{epoch+1}",
            )
            if fig is not None:
                writer.add_figure("Recon/Best_Batch", fig, epoch)
                plt.close(fig)
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:
                print(f"    [!] Early Stopping Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[-] Early stopping triggered after {epoch+1} epochs.")
            break

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vqvae_ep{epoch+1}.pth"))

    writer.close()
    
    # ================= 训练结束：执行重排序 =================
    print("\n" + "="*50)
    print("[*] Training Loop Finished. Starting Post-Training Codebook Optimization.")
    
    # 1. 加载最佳模型
    best_path = os.path.join(save_dir, "best_vqvae.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("    [-] Loaded best raw model.")
    
    # 2. 绘制排序前相似度
    plot_codebook_similarity(model, os.path.join(save_dir, "sim_matrix_raw.png"), title="Raw Codebook Similarity")
    
    # 3. 执行 PCA 排序 (已修正对小码本的支持)
    sort_codebook_by_pca(model)
    
    # 4. 绘制排序后相似度
    plot_codebook_similarity(model, os.path.join(save_dir, "sim_matrix_sorted.png"), title="Sorted Codebook Similarity")
    
    # 5. 保存
    sorted_save_path = os.path.join(save_dir, "best_vqvae_sorted.pth")
    torch.save(model.state_dict(), sorted_save_path)
    
    print(f"\n[*] Sorted model saved to: {sorted_save_path}")
    print("[!] IMPORTANT: Please use 'best_vqvae_sorted.pth' for downstream GPT training.")
    print("="*50)


if __name__ == "__main__":
    main()