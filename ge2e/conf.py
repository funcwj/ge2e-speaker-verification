# nnet opts

lstm_conf = {"num_layers": 3, "hidden_size": 738, "dropout": 0.2}

nnet_conf = {"feature_dim": 40, "embedding_dim": 256, "lstm_conf": lstm_conf}

# trainer opts
opt_kwargs = {"lr": 1e-2, "weight_decay": 1e-5, "momentum": 0.8}

trainer_conf = {
    "optimizer": "sgd",
    "optimizer_kwargs": opt_kwargs,
    "clip_norm": 10,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "no_impr": 6,
    "logging_period": 200  # steps
}

train_dir = "data/train"
dev_dir = "data/dev"