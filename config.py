import os

CONFIG = {
    # ==============================
    # 1. 数据与设备
    # ==============================
    "data_paths": [
        "/home/wujiaxuan/TS_LLM/data/output_kline/csi300.csv",
    ],
    
    "tokenizer_data_path": [
        "/home/wujiaxuan/TS_LLM/data/output_kline/csi300.csv",
    ],

    "val_data_path": "/home/wujiaxuan/TS_LLM/data/output_kline/csi300.csv",
    "index_data_path": "/home/wujiaxuan/TS_LLM/data/output_index_data/csi300_index.csv",
    "device": "cuda:5",
    "split_date": "2025-01-01",

    # ==============================
    # 2. 归一化 (金融序列的关键)
    # ==============================
    "norm_window": 60,
    "norm_eps": 1e-6,
    "norm_clip": 5.0, # 防止离群值毁掉 Codebook 训练

    # ==============================
    # 3. Tokenizer: RQ-VAE
    # ==============================
    "window_size": 32,
    "stride": 1,          
    "input_channels": 5,    
    "hidden_dim": 128,
    "n_embeddings": [512, 512], # 采用变长码本，浅层负责宏观，深层负责细节
    "embedding_dim": 64,
    "rvq_layers": 2,
    "commitment_cost": 0.25,
    "vq_decay": 0.99,
    "compression_rate": 8,

    "vqvae_batch_size": 512,
    "vqvae_lr": 3e-4,
    "vqvae_epochs": 100,
    "vqvae_save_dir": "./rqvae_ohlcv_800_long",
    "num_workers": 4,

    # Latent refine
    "use_latent_resblocks": True,
    "n_latent_resblocks": 2,
    "latent_dropout": 0.0,

    # ==============================
    # 4. Downstream: Time-LLM / GPT-2
    # ==============================
    "gpt_backbone": "/home/wujiaxuan/TS_LLM/GPT-2/gpt2-small",
    "n_gpt_layers": 6,
    "gpt_context_len": 32, 
    "thought_len": 10,
    "gpt_batch_size": 256, 
    
    "gpt_lr": 1e-4,           
    "gpt_weight_decay": 1e-3,
    "gpt_dropout": 0.1,
    "gpt_epochs": 20,         
    "gpt_save_dir": "./gpt_csi300_long", 
    "unfreeze_last_n_layers": 0, 
    
    # 损失权重：首层决定了大方向，给予较小权重避免过度拟合趋势的“噪音”
    "ce_layer_weights" : [1.0, 1.0], 
    "soft_label_sigma" : 3.0,
    
    # Trend Loss (信号回归增强)
    "trend_loss_weight": 0.0, # 适当调高以增强 RankIC 表现
    "trend_loss_start_epoch": 3,
    "trend_loss_warmup_epochs": 5,
    "pred_horizon": 8,        

    "seq_stride_days_train": 1,
    "seq_stride_days_val": 1,

    # ==============================
    # 6. 多模态与上下文配置
    # ==============================
    "num_stocks_est": 6000,
    "use_market_context": True,
}

os.makedirs(CONFIG["vqvae_save_dir"], exist_ok=True)
os.makedirs(CONFIG["gpt_save_dir"], exist_ok=True)