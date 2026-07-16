import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

# =============================================================================
# 1. 基础组件 (Projector, Fusion, Head)
# =============================================================================

class SimpleMLPProjector(nn.Module):
    """
    将拼接后的多层 RVQ Embedding 映射到 LLM 的维度
    """
    def __init__(self, n_rvq_layers, emb_dim, d_llm=768, dropout=0.1):
        super().__init__()
        input_dim = n_rvq_layers * emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm, d_llm), # 增加一层
            nn.GELU(),               # 增加激活
            nn.LayerNorm(d_llm)
        )

    def forward(self, x_embeds_list):
        # x_embeds_list: List of [B, T, Emb_Dim]
        x_concat = torch.cat(x_embeds_list, dim=-1)  # [B, T, L * Emb_Dim]
        return self.net(x_concat)


class GatedContextFusion(nn.Module):
    """
    门控融合模块：用于融合个股特征与大盘/行业上下文
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate_net = nn.Linear(d_model, d_model)

        # 保护机制：缩小权重，初始化 Bias=-2.0 (Sigmoid(-2) ≈ 0.11)
        nn.init.xavier_uniform_(self.gate_net.weight, gain=0.01)
        nn.init.constant_(self.gate_net.bias, -2.0)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_stock, x_context):
        context_feat = self.context_proj(x_context)
        gate = torch.sigmoid(self.gate_net(x_stock))
        fused = x_stock + gate * context_feat
        return self.norm(fused)


class CascadedRVQHead(nn.Module):
    """
    支持 Weight Tying 的级联解码头
    兼容每一层不同的 codebook size（变长码本）
    """
    def __init__(self, d_llm, n_rvq_layers, emb_dim):
        super().__init__()
        self.n_rvq_layers = n_rvq_layers
        self.emb_dim = emb_dim

        self.projections = nn.ModuleList()
        for i in range(n_rvq_layers):
            input_dim = d_llm if i == 0 else (d_llm + emb_dim)
            self.projections.append(nn.Sequential(
                nn.Linear(input_dim, d_llm),
                nn.GELU(),
                nn.LayerNorm(d_llm),
                nn.Linear(d_llm, emb_dim),
                nn.LayerNorm(emb_dim)
            ))

    def forward(self, hidden_state, codebook_layers, target_ids=None):
        """
        Args:
            hidden_state: [B, T, D_LLM]
            codebook_layers: nn.ModuleList(nn.Embedding) 用于 Weight Tying（每层可不同 vocab size）
            target_ids: [B, T, L] Teacher Forcing (可选)
        Returns:
            logits_list: List of [B, T, K_i]，其中 K_i 是第 i 层 codebook size
        """
        logits_list = []
        prev_emb = None

        for i in range(self.n_rvq_layers):
            if i == 0:
                proj_input = hidden_state
            else:
                proj_input = torch.cat([hidden_state, prev_emb], dim=-1)

            query_feat = self.projections[i](proj_input)

            # Weight tying: logits = query @ W^T
            logits = F.linear(query_feat, codebook_layers[i].weight)  # [B, T, K_i]
            logits_list.append(logits)

            if i < self.n_rvq_layers - 1:
                if target_ids is not None:
                    current_ids = target_ids[:, :, i]
                else:
                    current_ids = torch.argmax(logits, dim=-1)
                prev_emb = codebook_layers[i](current_ids)

        return logits_list


# =============================================================================
# 2. 增强版主模型 (AdaLN Stats Injection + Zero-Init Gated Fusion)
# =============================================================================

class TimeLLM_Enhanced(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # --- A. Backbone ---
        model_name = cfg.get("gpt_backbone", "gpt2")
        self.hf_config = GPT2Config.from_pretrained(model_name)
        target_layers = int(cfg.get("n_gpt_layers", 6))

        full_gpt = GPT2Model.from_pretrained(model_name, config=self.hf_config)
        if target_layers < self.hf_config.n_layer:
            full_gpt.h = full_gpt.h[:target_layers]
            full_gpt.config.n_layer = target_layers
            self.hf_config.n_layer = target_layers

        self.backbone = full_gpt

        # 1. 首先默认冻结所有参数
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. 解冻 wpe (权重位置编码)
        # 允许模型重新学习金融时间序列的时间演进逻辑，而不是复用文本的语法顺序
        self.backbone.wpe.weight.requires_grad = True

        # 3. 解冻 ln_f (最后的 LayerNorm 层)
        # 它是特征进入分类头之前的最后一道门槛，解冻它可以微调特征分布的缩放与位移
        self.backbone.ln_f.weight.requires_grad = True
        self.backbone.ln_f.bias.requires_grad = True

        # 如果你同时还需要解冻最后几层 Transformer Block
        unfreeze_last_n = int(cfg.get("unfreeze_last_n_layers", 0))
        if unfreeze_last_n > 0:
            total_layers = len(self.backbone.h)
            for i in range(total_layers - unfreeze_last_n, total_layers):
                for param in self.backbone.h[i].parameters():
                    param.requires_grad = True

        self.d_llm = int(self.hf_config.n_embd)
        self.max_pos_embeddings = self.hf_config.n_positions
        self.gpt_context_len = int(cfg.get("gpt_context_len", 32))

        # --- B. 输入配置 ---
        self.n_rvq_layers = int(cfg.get("rvq_layers", 4))
        self.emb_dim = int(cfg["embedding_dim"])
        gpt_dropout = float(cfg.get("gpt_dropout", 0.1))

        # --- [关键修改] 解析变长码本配置 n_embeddings ---
        n_embeddings_config = cfg["n_embeddings"]
        if isinstance(n_embeddings_config, str):
            try:
                import ast
                n_embeddings_config = ast.literal_eval(n_embeddings_config)
            except Exception:
                pass

        if isinstance(n_embeddings_config, int) or (isinstance(n_embeddings_config, str) and n_embeddings_config.isdigit()):
            num_embeddings_list = [int(n_embeddings_config)] * self.n_rvq_layers
        elif isinstance(n_embeddings_config, (list, tuple)):
            num_embeddings_list = list(n_embeddings_config)
        else:
            raise ValueError(f"Invalid format for n_embeddings: {n_embeddings_config}")

        # 长度对齐（与 rqvae 一致的安全处理）
        if len(num_embeddings_list) != self.n_rvq_layers:
            if len(num_embeddings_list) > self.n_rvq_layers:
                num_embeddings_list = num_embeddings_list[:self.n_rvq_layers]
            else:
                num_embeddings_list += [num_embeddings_list[-1]] * (self.n_rvq_layers - len(num_embeddings_list))

        self.num_embeddings_list = num_embeddings_list  # 保存，生成时也会用到

        # Embedding Layers（每层不同 vocab size）
        self.source_embeds = nn.ModuleList(
            [nn.Embedding(self.num_embeddings_list[i], self.emb_dim) for i in range(self.n_rvq_layers)]
        )

        # 冻结 Codebook 参数，避免语义漂移
        for emb in self.source_embeds:
            emb.weight.requires_grad = False

        self.projector = SimpleMLPProjector(
            n_rvq_layers=self.n_rvq_layers,
            emb_dim=self.emb_dim,
            d_llm=self.d_llm,
            dropout=gpt_dropout
        )

        # --- C. 上下文融合 ---
        self.use_market_context = cfg.get("use_market_context", False)
        if self.use_market_context:
            self.market_source_embeds = nn.ModuleList(
                [nn.Embedding(self.num_embeddings_list[i], self.emb_dim) for i in range(self.n_rvq_layers)]
            )
            for emb in self.market_source_embeds:
                emb.weight.requires_grad = False

            self.market_projector = SimpleMLPProjector(
                n_rvq_layers=self.n_rvq_layers,
                emb_dim=self.emb_dim,
                d_llm=self.d_llm,
                dropout=gpt_dropout
            )
            self.fusion_layer = GatedContextFusion(self.d_llm, dropout=gpt_dropout)

        # --- D. 时间特征与门控 ---
        self.day_embed = nn.Embedding(32, self.d_llm)
        self.month_embed = nn.Embedding(13, self.d_llm)
        self.weekday_embed = nn.Embedding(7, self.d_llm)
        self.time_gates = nn.Parameter(torch.zeros(3))

        # --- E. 统计特征调制 (AdaLN) ---
        self.stats_input_dim = int(cfg.get("stats_input_dim", 5))
        self.embed_ln = nn.LayerNorm(self.d_llm, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.stats_input_dim, self.d_llm * 2)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # --- F. Prompt ---
        self.thought_len = int(cfg.get("thought_len", 5))
        self.thought_embed = nn.Parameter(torch.randn(1, self.thought_len, self.d_llm))
        nn.init.normal_(self.thought_embed, std=0.02)

        # --- G. 输出头 ---
        self.cascaded_head = CascadedRVQHead(
            d_llm=self.d_llm,
            n_rvq_layers=self.n_rvq_layers,
            emb_dim=self.emb_dim
        )

    def forward(self, input_ids, market_ids=None, time_feats=None, target_ids=None, stats_feats=None):
        """
        Args:
            input_ids: [B, T, L]
            market_ids: [B, T, L] (Optional)
            time_feats: [B, T, 3] (Optional)
            target_ids: [B, T, L] (Optional)
            stats_feats: [B, C] (Optional)
        """
        B, T, L = input_ids.shape
        if L != self.n_rvq_layers:
            raise ValueError(f"input_ids last dim (L={L}) must equal rvq_layers (={self.n_rvq_layers})")

        # 1. 个股特征 Embedding
        raw_embeds = [self.source_embeds[i](input_ids[:, :, i]) for i in range(self.n_rvq_layers)]
        x = self.projector(raw_embeds)

        # 2. 上下文融合
        if self.use_market_context and market_ids is not None:
            if market_ids.shape[-1] != self.n_rvq_layers:
                raise ValueError("market_ids last dim must equal rvq_layers")
            market_raw = [self.market_source_embeds[i](market_ids[:, :, i]) for i in range(self.n_rvq_layers)]
            x_market = self.market_projector(market_raw)
            x = self.fusion_layer(x_stock=x, x_context=x_market)

        # 3. 时间特征注入（零初始化门控）
        if time_feats is not None:
            d_emb = self.day_embed(time_feats[:, :, 0])
            m_emb = self.month_embed(time_feats[:, :, 1])
            w_emb = self.weekday_embed(time_feats[:, :, 2])
            time_context = (self.time_gates[0] * d_emb) + \
                           (self.time_gates[1] * m_emb) + \
                           (self.time_gates[2] * w_emb)
            x = x + time_context

        # 4. AdaLN 调制
        if stats_feats is not None:
            shift_scale = self.adaLN_modulation(stats_feats)
            shift, scale = shift_scale.chunk(2, dim=-1)
            x = self.embed_ln(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            x = self.embed_ln(x)

        # 5. 拼接 Prompt
        thoughts = self.thought_embed.expand(B, -1, -1)

        max_input_len = self.max_pos_embeddings - self.thought_len
        if x.shape[1] > max_input_len:
            x = x[:, -max_input_len:, :]
            if target_ids is not None:
                target_ids = target_ids[:, -max_input_len:, :]

        final_inputs = torch.cat([thoughts, x], dim=1)

        # 6. Backbone Forward
        outputs = self.backbone(inputs_embeds=final_inputs)
        hidden = outputs.last_hidden_state[:, self.thought_len:, :]

        # 7. Decoding（每层 logits 的最后一维不同）
        logits_list = self.cascaded_head(hidden, self.source_embeds, target_ids=target_ids)

        return logits_list

    @torch.no_grad()
    def generate(self, context_ids, steps, temperature=1.0, top_k=0, top_p=0.9,
                 market_ids=None, time_feats=None, stats_feats=None):
        """
        生成函数：支持变长码本（每层 vocab size 不同）
        """
        self.eval()
        curr_ids = context_ids

        for _ in range(steps):
            if curr_ids.shape[1] > self.gpt_context_len:
                model_input = curr_ids[:, -self.gpt_context_len:, :]
            else:
                model_input = curr_ids

            curr_len = model_input.shape[1]

            if market_ids is not None:
                curr_market = market_ids[:, -curr_len:, :]
            else:
                curr_market = None

            if time_feats is not None:
                curr_time = time_feats[:, -curr_len:, :]
            else:
                curr_time = None

            # (A) Embed & Project
            raw_embeds = [self.source_embeds[i](model_input[:, :, i]) for i in range(self.n_rvq_layers)]
            x = self.projector(raw_embeds)

            # (B) Fusion
            if self.use_market_context and curr_market is not None:
                market_raw = [self.market_source_embeds[i](curr_market[:, :, i]) for i in range(self.n_rvq_layers)]
                x_market = self.market_projector(market_raw)
                x = self.fusion_layer(x, x_market)

            # (C) Time Feats
            if curr_time is not None:
                d_emb = self.day_embed(curr_time[:, :, 0])
                m_emb = self.month_embed(curr_time[:, :, 1])
                w_emb = self.weekday_embed(curr_time[:, :, 2])
                time_context = (self.time_gates[0] * d_emb) + \
                               (self.time_gates[1] * m_emb) + \
                               (self.time_gates[2] * w_emb)
                x = x + time_context

            # (D) Stats Injection (AdaLN)
            if stats_feats is not None:
                shift_scale = self.adaLN_modulation(stats_feats)
                shift, scale = shift_scale.chunk(2, dim=-1)
                x = self.embed_ln(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            else:
                x = self.embed_ln(x)

            # (E) Backbone
            B = curr_ids.shape[0]
            thoughts = self.thought_embed.expand(B, -1, -1)
            final_inputs = torch.cat([thoughts, x], dim=1)
            outputs = self.backbone(inputs_embeds=final_inputs)
            hidden = outputs.last_hidden_state[:, self.thought_len:, :]
            last_hidden = hidden[:, -1:, :]  # [B, 1, d_llm]

            # (F) Cascaded Decoding（逐层采样，逐层 vocab size 不同）
            next_token_layers = []
            prev_emb = None

            for i in range(self.n_rvq_layers):
                if i == 0:
                    proj_input = last_hidden
                else:
                    proj_input = torch.cat([last_hidden, prev_emb], dim=-1)

                query_feat = self.cascaded_head.projections[i](proj_input)  # [B, 1, emb_dim]
                logits = F.linear(query_feat, self.source_embeds[i].weight)  # [B, 1, K_i]
                next_token_logits = logits[:, -1, :] / float(temperature)     # [B, K_i]

                filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)          # [B, 1], long
                next_token_layers.append(next_token)

                if i < self.n_rvq_layers - 1:
                    prev_emb = self.source_embeds[i](next_token)              # [B, 1, emb_dim]

            # [B, L] -> [B, 1, L]
            next_step = torch.cat(next_token_layers, dim=1).unsqueeze(1)
            curr_ids = torch.cat([curr_ids, next_step], dim=1)

        return curr_ids[:, context_ids.shape[1]:, :]

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
        if top_k > 0:
            kth = torch.topk(logits, top_k)[0][..., -1, None]
            indices_to_remove = logits < kth
            logits = logits.clone()
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.clone()
            logits[indices_to_remove] = filter_value

        return logits