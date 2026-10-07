from pathlib import Path

import yaml

from .io import digest


ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    config = yaml.safe_load(Path(path or ROOT / "configs/study.yaml").read_text())
    t = config["training"]
    if t["scale_rewards"] != "none" or t["loss_type"] != "dr_grpo":
        raise ValueError("The registered protocol requires unscaled rewards and dr_grpo.")
    if set(config["thresholds"]["primary"]) & set(config["thresholds"]["train"]):
        raise ValueError("Primary thresholds must be held out from training.")
    for ts in config["thresholds"].values():
        if not all(0 < x < 1 for x in ts):
            raise ValueError("All thresholds must be strictly between zero and one.")
    batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    if batch % t["num_generations"]:
        raise ValueError("Effective completion batch must be divisible by generations.")
    if t["steps_per_generation"] != t["gradient_accumulation_steps"]:
        raise ValueError("Protocol requires one generation batch per optimizer update.")
    for n in (config["data"]["train"], config["data"]["reduced_train"]):
        if n % (batch // t["num_generations"]):
            raise ValueError("Training size must contain complete question batches.")
    return config


def fingerprint(config):
    return digest(config)


def matrix(config):
    for model in config["models"]:
        arms = ["binary", "conditioned"] + (["fixed"] if model == "qwen" else [])
        for seed in config["seeds"]:
            for arm in arms:
                yield {"model": model, "arm": arm, "seed": seed,
                       "run_id": f"{model}-{arm}-s{seed}"}
