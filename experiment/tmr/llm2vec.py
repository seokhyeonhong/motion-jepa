"""Frozen Llama-3 LLM2Vec sentence features for Kimodo-style TMR.

Uses the MNTP adapter followed by the supervised adapter, bidirectional
attention, Llama-3 user formatting and instruction-excluding mean pooling.
The explicit 4D mask works with the pinned Transformers 5.1 implementation;
the older LLM2Vec _update_causal_mask hook is no longer called there.
"""

from __future__ import annotations

import torch
from torch import nn


class LLM2VecBackbone(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.feature_dim = int(model.config.hidden_size)

    def prepare_text(self, text, max_length):
        # Match LLM2Vec's document limit before adding the user template.
        while len(self.tokenizer(
            text, truncation=True, max_length=max_length, add_special_tokens=False,
        )["input_ids"]) > 400:
            count = len(self.tokenizer(
                text, truncation=True, max_length=max_length, add_special_tokens=False,
            )["input_ids"])
            text = " ".join(text.split()[:int(len(text.split()) * 400 / count)])
        content = text.strip() + "<|eot_id|>"
        return "<|start_header_id|>user<|end_header_id|>\n\n" + content, content

    def token_length(self, text, max_length):
        formatted, _ = self.prepare_text(text, max_length)
        return len(self.tokenizer(formatted, truncation=False)["input_ids"])

    def hidden_states(self, input_ids, valid_mask):
        # A supplied 4D additive mask bypasses causal-mask creation in HF.
        dtype = self.model.get_input_embeddings().weight.dtype
        attention_mask = torch.zeros(
            (len(input_ids), 1, input_ids.shape[1], input_ids.shape[1]),
            dtype=dtype, device=input_ids.device,
        ).masked_fill(~valid_mask[:, None, None, :].bool(), torch.finfo(dtype).min)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state

    @torch.inference_mode()
    def encode(self, texts, max_length, device):
        # Kimodo uses one caption per forward pass for BF16 reproducibility.
        vectors = []
        for text in texts:
            formatted, content = self.prepare_text(text, max_length)
            inputs = self.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=max_length,
            ).to(device)
            content_length = len(self.tokenizer(
                content, truncation=True, max_length=max_length, add_special_tokens=False,
            )["input_ids"])
            hidden = self.hidden_states(inputs["input_ids"], inputs["attention_mask"])
            vectors.append(hidden[0, -content_length:].mean(dim=0, keepdim=True).cpu())
        return vectors


def load_llm2vec(model_name, revision, adapter_name, adapter_revision, device):
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoTokenizer, LlamaModel
    from transformers.utils.hub import cached_file, extract_commit_hash

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    if config.model_type != "llama" or config.hidden_size != 4096:
        raise ValueError("LLM2Vec requires a Llama-3 4096D MNTP checkpoint")
    resolved_revision = getattr(config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    mntp = PeftConfig.from_pretrained(model_name, revision=resolved_revision)
    base_config = AutoConfig.from_pretrained(mntp.base_model_name_or_path, revision=mntp.revision)
    base_revision = getattr(base_config, "_commit_hash", None) or mntp.revision
    model, loading = LlamaModel.from_pretrained(
        mntp.base_model_name_or_path, revision=base_revision,
        config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"LLM2Vec base weights did not load completely: {loading}")
    model = model.to(device)
    model = PeftModel.from_pretrained(model, model_name, revision=resolved_revision)
    model = model.merge_and_unload()
    adapter_file = cached_file(adapter_name, "adapter_config.json", revision=adapter_revision)
    resolved_adapter_revision = extract_commit_hash(adapter_file, None) or adapter_revision
    model = PeftModel.from_pretrained(model, adapter_name, revision=resolved_adapter_revision)
    model = model.to(device).eval().requires_grad_(False)
    backbone = LLM2VecBackbone(model, tokenizer).eval().requires_grad_(False)
    return backbone, tokenizer, {
        "model": str(model_name), "revision": revision,
        "resolved_revision": resolved_revision,
        "base_model": mntp.base_model_name_or_path, "base_revision": base_revision,
        "adapter": str(adapter_name), "adapter_revision": adapter_revision,
        "resolved_adapter_revision": resolved_adapter_revision,
        "feature_dim": 4096, "dtype": "bfloat16", "model_class": "LLM2VecBackbone",
        "max_position_embeddings": int(config.max_position_embeddings),
        "attention": "bidirectional", "pooling": "mean_without_instruction",
        "extraction_batch_size": 1,
    }
