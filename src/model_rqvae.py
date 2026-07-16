import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Vector Quantizer (EMA Version) - 基础量化层
# =============================================================================
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: [B, C=emb_dim, T=latent_len]
        # 转换维度以便计算距离: [B, T, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input: [B*T, C]
        flat_input = inputs.view(-1, self._embedding_dim)

        # 计算距离: (a-b)^2 = a^2 + b^2 - 2ab
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding: 找到最近的 Embedding Index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*T, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize: 根据 Index 取出向量
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)  # [B, T, C]

        # EMA 更新逻辑 (仅在训练时进行)
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing to prevent division by zero
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)  # [K, C]
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            # 更新 Embedding 权重
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss 计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator (STE)
        quantized = inputs + (quantized - inputs).detach()
        
        # 返回 loss, quantized [B, C, T], indices [B, T]
        return loss, quantized.permute(0, 2, 1).contiguous(), encoding_indices.view(input_shape[0], -1)


# =============================================================================
# 2. Residual VQ - 支持变长码本
# =============================================================================
class ResidualVQ(nn.Module):
    def __init__(self, num_embeddings_list, embedding_dim, n_layers=4, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.n_layers = n_layers
        
        # [修改] 兼容处理：如果传入的是 int，则转为 list；如果是 list，则校验长度
        if isinstance(num_embeddings_list, int):
            num_embeddings_list = [num_embeddings_list] * n_layers
        
        # 安全检查：确保列表长度与 n_layers 一致
        if len(num_embeddings_list) != n_layers:
            # 如果配置给的列表长度不对，这里做一个简单的截断或填充处理，防止报错
            if len(num_embeddings_list) > n_layers:
                num_embeddings_list = num_embeddings_list[:n_layers]
            else:
                # 不足的部分用最后一层的大小填充
                num_embeddings_list += [num_embeddings_list[-1]] * (n_layers - len(num_embeddings_list))

        # [修改] 逐层使用不同的 Codebook Size 初始化
        self.layers = nn.ModuleList(
            [
                VectorQuantizerEMA(
                    num_embeddings=num_embeddings_list[i], 
                    embedding_dim=embedding_dim, 
                    commitment_cost=commitment_cost, 
                    decay=decay
                ) 
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        quantized_out = 0.0
        residual = x
        total_loss = 0.0
        all_indices = []

        for layer in self.layers:
            loss, quantized, indices = layer(residual)
            quantized_out = quantized_out + quantized
            
            # 计算残差：当前残差 - 当前量化值 = 下一层残差
            residual = residual - quantized.detach()
            
            total_loss += loss
            all_indices.append(indices)

        # 堆叠所有层的索引: [B, T, n_layers]
        codes = torch.stack(all_indices, dim=-1)
        
        return total_loss, quantized_out, codes


# =============================================================================
# 3. VQVAE Daily - 主模型
# =============================================================================
class VQVAE_Daily(nn.Module):
    """
    Conv1d stride=2 两次 => latent_len = window_size / 4
    ConvTranspose1d stride=2 两次 => recon_len = latent_len * 4
    因此 window_size 必须能被 4 整除。
    """

    def __init__(self, cfg):
        super().__init__()
        in_dim = int(cfg["input_channels"])
        h_dim = int(cfg["hidden_dim"])
        emb_dim = int(cfg["embedding_dim"])
        n_layers = int(cfg.get("rvq_layers", 4))
        window_size = int(cfg["window_size"])

        if window_size % 4 != 0:
            raise ValueError(f"window_size must be divisible by 4, got {window_size}")

        # --- Encoder ---
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_dim, h_dim, 4, 2, 1),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(h_dim, h_dim, 4, 2, 1),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1),
        )

        # --- Pre-quant Projection ---
        self.pre_quant = nn.Sequential(
            nn.Conv1d(h_dim, emb_dim, 1),
            nn.BatchNorm1d(emb_dim),
        )

        # --- [修改] 解析 n_embeddings 配置 ---
        n_embeddings_config = cfg["n_embeddings"]
        # 如果配置是字符串（例如命令行参数传入时），尝试解析
        if isinstance(n_embeddings_config, str):
            try:
                # 尝试解析类似于 "[1024, 512]" 的字符串
                import ast
                n_embeddings_config = ast.literal_eval(n_embeddings_config)
            except:
                pass # 保持原样，可能是单个数字的字符串

        if isinstance(n_embeddings_config, int) or (isinstance(n_embeddings_config, str) and n_embeddings_config.isdigit()):
             # 如果是单一整数
             num_embeddings_list = [int(n_embeddings_config)] * n_layers
        elif isinstance(n_embeddings_config, (list, tuple)):
             # 如果是列表
             num_embeddings_list = list(n_embeddings_config)
        else:
             raise ValueError(f"Invalid format for n_embeddings: {n_embeddings_config}")

        # --- RVQ ---
        self.rvq = ResidualVQ(
            num_embeddings_list=num_embeddings_list, # 传入处理好的列表
            embedding_dim=emb_dim,
            n_layers=n_layers,
            commitment_cost=float(cfg["commitment_cost"]),
            decay=float(cfg.get("vq_decay", 0.99)),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(emb_dim, h_dim, 4, 2, 1),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(h_dim, h_dim, 4, 2, 1),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(h_dim, in_dim, 3, 1, 1),
        )

    def forward(self, x):
        # x: [B, C, window_size]
        h = self.encoder_conv(x)
        z = self.pre_quant(h)
        loss_vq, quantized, codes = self.rvq(z)
        x_recon = self.decoder(quantized)
        return loss_vq, x_recon, codes

    def encode(self, x):
        """
        仅用于推理或生成数据集时提取 Codes
        """
        h = self.encoder_conv(x)
        z = self.pre_quant(h)
        _, _, codes = self.rvq(z)
        return codes