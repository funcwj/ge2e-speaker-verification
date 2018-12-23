# nnet opts

lstm_conf = {"num_layers": 3, "hidden_size": 738, "dropout": 0.5}

nnet_conf = {"feature_dim": 40, "embedding_dim": 256, "lstm_conf": lstm_conf}

# trainer opts
opt_kwargs = {"lr": 1e-2, "weight_decay": 1e-5, "momentum": 0.8}

trainer_conf = {
    "optimizer": "sgd",
    "optimizer_kwargs": opt_kwargs,
    "clip_norm": 3,
    "min_lr": 1e-8,
    "patience": 3,
    "factor": 0.5,
    "logging_period": 1000  # steps
}

# loader opts

train_steps = 200000
dev_steps = 10000
chunk_size = (140, 180)