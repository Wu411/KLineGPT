import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Vector Quantizer (EMA Version) - 基础量化层
# =============================================================================
class VectorQuantizerEMA(nn.Module):
    """
    Stable EMA VQ:
    - embedding is updated by EMA, no gradient on embedding.weight
    - EMA states are buffers, updated in-place
    - forward returns:
        loss: scalar
        quantized: [B, T, C]  (same as your current interface)
        indices:  [B, T]
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_l2_normalize: bool = False,   # optional; keep False to match your Euclidean setup
    ):
        super().__init__()
        self._embedding_dim = int(embedding_dim)
        self._num_embeddings = int(num_embeddings)
        self._commitment_cost = float(commitment_cost)
        self._decay = float(decay)
        self._epsilon = float(epsilon)
        self._use_l2_normalize = bool(use_l2_normalize)

        # Codebook (EMA-updated, no grad)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._embedding.weight.requires_grad_(False)

        # EMA buffers
        self.register_buffer("_ema_cluster_size", torch.zeros(self._num_embeddings))
        self.register_buffer("_ema_w", torch.zeros(self._num_embeddings, self._embedding_dim))
        self._ema_w.data.copy_(self._embedding.weight.data)

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, C, T] -> [B, T, C]
        if inputs.dim() != 3 or inputs.size(1) != self._embedding_dim:
            raise ValueError(
                f"Expected inputs [B, C={self._embedding_dim}, T], got {tuple(inputs.shape)}"
            )

        x = inputs.permute(0, 2, 1).contiguous()   # [B, T, C]
        B, T, C = x.shape

        flat_x = x.view(-1, C)                     # [B*T, C]

        # optional normalize (off by default)
        if self._use_l2_normalize:
            flat_x_q = F.normalize(flat_x, p=2, dim=-1)
            codebook = F.normalize(self._embedding.weight, p=2, dim=-1)
        else:
            flat_x_q = flat_x
            codebook = self._embedding.weight

        # distances: ||x||^2 + ||e||^2 - 2 x e^T
        # [N, 1] + [K] - 2[N,K] -> [N,K]
        x2 = torch.sum(flat_x_q ** 2, dim=1, keepdim=True)            # [N,1]
        e2 = torch.sum(codebook ** 2, dim=1).unsqueeze(0)             # [1,K]
        xe = torch.matmul(flat_x_q, codebook.t())                     # [N,K]
        distances = x2 + e2 - 2.0 * xe

        # nearest code
        encoding_indices = torch.argmin(distances, dim=1)             # [N]
        encodings = F.one_hot(encoding_indices, self._num_embeddings).type(flat_x.dtype)  # [N,K]

        # quantized vectors
        quantized = torch.matmul(encodings, codebook).view(B, T, C)   # [B,T,C]

        # EMA update (train only)
        if self.training:
            with torch.no_grad():
                # counts
                cluster_size = encodings.sum(dim=0)                   # [K]
                self._ema_cluster_size.mul_(self._decay).add_(cluster_size, alpha=1.0 - self._decay)

                # Laplace smoothing
                n = self._ema_cluster_size.sum()
                smoothed = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon)
                    * n
                )

                # codebook sums
                dw = torch.matmul(encodings.t(), flat_x_q)            # [K,C]
                self._ema_w.mul_(self._decay).add_(dw, alpha=1.0 - self._decay)

                # normalized mean
                new_weight = self._ema_w / smoothed.unsqueeze(1)      # [K,C]
                if self._use_l2_normalize:
                    new_weight = F.normalize(new_weight, p=2, dim=-1)

                # in-place update of embedding weights
                self._embedding.weight.data.copy_(new_weight)

        # commitment loss (match your design: quantized detached vs inputs)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self._commitment_cost * e_latent_loss

        # straight-through estimator
        quantized_st = x + (quantized - x).detach()                   # [B,T,C]

        # return indices shaped [B,T]
        indices_bt = encoding_indices.view(B, T)

        return loss, quantized_st, indices_bt


# =============================================================================
# 2. Residual VQ - 支持变长码本
# =============================================================================
class ResidualVQ(nn.Module):
    def __init__(self, num_embeddings_list, embedding_dim, n_layers=4, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.n_layers = int(n_layers)

        # 兼容：int -> list
        if isinstance(num_embeddings_list, int):
            num_embeddings_list = [int(num_embeddings_list)] * self.n_layers

        # 安全检查：长度对齐
        if len(num_embeddings_list) != self.n_layers:
            if len(num_embeddings_list) > self.n_layers:
                num_embeddings_list = num_embeddings_list[: self.n_layers]
            else:
                num_embeddings_list = list(num_embeddings_list) + [int(num_embeddings_list[-1])] * (self.n_layers - len(num_embeddings_list))

        self.layers = nn.ModuleList(
            [
                VectorQuantizerEMA(
                    num_embeddings=int(num_embeddings_list[i]),
                    embedding_dim=int(embedding_dim),
                    commitment_cost=float(commitment_cost),
                    decay=float(decay),
                )
                for i in range(self.n_layers)
            ]
        )

    def forward(self, x):
        # x: [B, C, T]
        quantized_out = 0.0
        residual = x
        total_loss = 0.0
        all_indices = []

        for layer in self.layers:
            loss, quantized, indices = layer(residual)   # quantized: [B, T, C]
            # 把 quantized 转回 [B, C, T] 以便残差计算和 decoder 使用
            quantized_ct = quantized.permute(0, 2, 1).contiguous()  # [B, C, T]

            quantized_out = quantized_out + quantized_ct
            residual = residual - quantized_ct.detach()

            total_loss += loss
            all_indices.append(indices)  # [B, T]

        # codes: [B, T, n_layers]
        codes = torch.stack(all_indices, dim=-1)
        return total_loss, quantized_out, codes


# =============================================================================
# 3. Latent Residual Block - 用于增强重构（不改变 token 数）
# =============================================================================
class ResBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(x + self.net(x))


# =============================================================================
# 4. VQVAE Daily - 主模型 (1 token = 8 天)
# =============================================================================
class VQVAE_Daily(nn.Module):
    """
    通过 compression_rate 控制 1 token 对应的天数（默认 8）。

    - compression_rate = 8  -> stride=2 做 3 次 -> latent_len = window_size / 8
    - Decoder 对称上采样 3 次 -> recon_len = latent_len * 8
    """

    def __init__(self, cfg):
        super().__init__()
        in_dim = int(cfg["input_channels"])
        h_dim = int(cfg["hidden_dim"])
        emb_dim = int(cfg["embedding_dim"])
        n_layers = int(cfg.get("rvq_layers", 4))
        window_size = int(cfg["window_size"])

        # 关键：token 粒度
        compression_rate = int(cfg.get("compression_rate", 8))
        if compression_rate <= 0 or (compression_rate & (compression_rate - 1)) != 0:
            raise ValueError(f"compression_rate must be a power of 2, got {compression_rate}")
        n_down = int(math.log2(compression_rate))

        if window_size % compression_rate != 0:
            raise ValueError(
                f"window_size must be divisible by compression_rate={compression_rate}, "
                f"got window_size={window_size}"
            )

        self.compression_rate = compression_rate
        self.n_down = n_down

        # --------------------------
        # Encoder: stride=2 * n_down  (总倍率 = compression_rate)
        # --------------------------
        enc_layers = []
        ch_in = in_dim
        ch = h_dim
        for _ in range(n_down):
            enc_layers += [
                nn.Conv1d(ch_in, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.1),
            ]
            ch_in = ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Pre-quant projection
        self.pre_quant = nn.Sequential(
            nn.Conv1d(h_dim, emb_dim, 1),
            nn.BatchNorm1d(emb_dim),
        )

        # 可选：latent 细化模块（提高重构，不改变 token 数）
        use_latent_resblocks = bool(cfg.get("use_latent_resblocks", True))
        n_latent_resblocks = int(cfg.get("n_latent_resblocks", 2))
        latent_dropout = float(cfg.get("latent_dropout", 0.0))
        if use_latent_resblocks and n_latent_resblocks > 0:
            self.latent_refine = nn.Sequential(
                *[ResBlock1D(emb_dim, dropout=latent_dropout) for _ in range(n_latent_resblocks)]
            )
        else:
            self.latent_refine = nn.Identity()

        # 解析 n_embeddings（保持你原来的输入方式）
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

        # RVQ
        self.rvq = ResidualVQ(
            num_embeddings_list=num_embeddings_list,
            embedding_dim=emb_dim,
            n_layers=n_layers,
            commitment_cost=float(cfg["commitment_cost"]),
            decay=float(cfg.get("vq_decay", 0.99)),
        )

        # --------------------------
        # Decoder: stride=2 * n_down
        # --------------------------
        dec_layers = []
        ch_in = emb_dim
        ch = h_dim
        for _ in range(n_down):
            dec_layers += [
                nn.ConvTranspose1d(ch_in, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.1),
            ]
            ch_in = ch

        # 输出层
        dec_layers += [
            nn.ConvTranspose1d(h_dim, in_dim, kernel_size=3, stride=1, padding=1),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: [B, C, window_size]
        h = self.encoder_conv(x)      # [B, h_dim, window_size / compression_rate]
        z = self.pre_quant(h)         # [B, emb_dim, T_latent]
        z = self.latent_refine(z)     # [B, emb_dim, T_latent]

        loss_vq, quantized_ct, codes = self.rvq(z)  # quantized_ct: [B, emb_dim, T_latent]
        x_recon = self.decoder(quantized_ct)        # [B, C, window_size]

        return loss_vq, x_recon, codes

    def encode(self, x):
        """
        仅用于推理或生成数据集时提取 Codes
        返回 codes: [B, T_latent, n_layers]
        """
        h = self.encoder_conv(x)
        z = self.pre_quant(h)
        z = self.latent_refine(z)
        _, _, codes = self.rvq(z)
        return codes