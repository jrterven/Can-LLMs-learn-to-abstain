from __future__ import annotations

import math
import time
from pathlib import Path

from .io import read_json, utcnow, write_json
from .prompts import confidence_messages


def pin_models(config, workdir, download=False):
    from huggingface_hub import HfApi, snapshot_download
    path = Path(workdir) / "artifacts/models.lock.json"
    if path.exists():
        pins = read_json(path)
        for key, spec in config["models"].items():
            if pins[key]["repo"] != spec["repo"]:
                raise ValueError("Model identifiers changed after pinning.")
    else:
        api = HfApi()
        pins = {key: {"repo": spec["repo"], "revision": api.model_info(spec["repo"]).sha,
                      "pinned_at": utcnow()} for key, spec in config["models"].items()}
        write_json(path, pins)
    if download:
        for spec in pins.values():
            snapshot_download(spec["repo"], revision=spec["revision"],
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    return pins


def load_model(config, workdir, key, adapter=None, trainable=False, initializer=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for research runs; unit tests do not require one.")
    pin = read_json(Path(workdir) / "artifacts/models.lock.json")[key]
    tokenizer = AutoTokenizer.from_pretrained(pin["repo"], revision=pin["revision"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pin["repo"], revision=pin["revision"],
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    if initializer:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(initializer)).merge_and_unload()
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    model.eval()
    return model, tokenizer


def render(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def eos_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tokenizer.eos_token_id
    return set(ids if isinstance(ids, list) else [ids])


def is_truncated(token_ids, eos):
    return not any(token in eos for token in token_ids)


class Engine:
    def __init__(self, config, workdir, key, adapter=None, initializer=None):
        self.config, self.key = config, key
        self.model, self.tokenizer = load_model(config, workdir, key, adapter, initializer=initializer)
        self.eos = eos_ids(self.model, self.tokenizer)

    def generate(self, conversations, sample=False, temperature=None):
        import torch
        prompts = [render(self.tokenizer, m) for m in conversations]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        if encoded.input_ids.shape[1] > self.config["training"]["max_prompt_length"]:
            raise ValueError("Prompt exceeds registered limit; refusing silent truncation.")
        encoded = encoded.to(self.model.device)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            kwargs = {"do_sample": sample, "max_new_tokens": self.config["training"]["max_completion_length"],
                      "pad_token_id": self.tokenizer.pad_token_id, "use_cache": True}
            if sample:
                kwargs.update(temperature=temperature or self.config["training"]["temperature"], top_p=1.0, top_k=0)
            out = self.model.generate(**encoded, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        results = []
        for ids in out[:, encoded.input_ids.shape[1]:].tolist():
            stop = next((i + 1 for i, token in enumerate(ids) if token in self.eos), len(ids))
            ids = ids[:stop]
            results.append({"text": self.tokenizer.decode(ids, skip_special_tokens=True),
                            "truncated": is_truncated(ids, self.eos), "generated_tokens": len(ids),
                            "seconds": elapsed / len(prompts)})
        return results

    def confidence(self, question, answer):
        """Normalized sequence likelihood of True/False, not a verbal percentage."""
        import torch
        prompt = render(self.tokenizer, confidence_messages(question, answer))
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        endings = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("True", "False")]
        if len(prefix) + max(map(len, endings)) > self.config["training"]["max_prompt_length"]:
            raise ValueError("Confidence prompt exceeds registered token budget.")
        started = time.monotonic()
        scores = []
        with torch.inference_mode():
            for ending in endings:
                ids = torch.tensor([prefix + ending], device=self.model.device)
                logits = self.model(ids).logits[0, len(prefix)-1:-1].float().log_softmax(-1)
                score = logits[torch.arange(len(ending), device=ids.device), torch.tensor(ending, device=ids.device)].sum()
                scores.append(float(score))
        logit = scores[0] - scores[1]
        p_true = 1 / (1 + math.exp(-max(-700, min(700, logit))))
        torch.cuda.synchronize()
        return {"p_true": p_true, "confidence_logit": logit, "confidence_seconds": time.monotonic() - started}

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
