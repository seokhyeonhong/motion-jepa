"""Frozen JEPA frame-token readouts for TMR; attention pooling with optional RoPE blocks."""
from __future__ import annotations
import copy
import json
import os
import shutil
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.modules import TransformerBlock1D, initialize_transformer
from model.pos_embs import RotaryPosEmbed1D, temporal_token_positions
from . import layer_probe as probe, train
from .dataset import PreparedPairDataset, collate_pairs, json_digest
from .features import load_prepared_datasets
from .lora import load_motion_initialization
from .model import AlignmentConfig, _SequenceReadout
from .utils import resolve_device
from experiment.linear_probe.features import _atomic_json_save, _sha256_file

EXPERIMENT = "tmr_frozen_temporal_readout"

class TemporalReadout(nn.Module):
    def __init__(self, dim, depth=0, num_heads=6, fps=30., theta=100., time_scale=1.):
        super().__init__()
        if depth not in (0, 1, 2) or dim % num_heads or fps <= 0:
            raise ValueError("Invalid temporal readout specification")
        self.input_dim, self.fps = dim, fps
        self.linear = nn.Linear(dim, dim)
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_scale", torch.ones(dim))
        self.linear.apply(initialize_transformer)
        self.blocks = nn.ModuleList([
            TransformerBlock1D(dim, num_heads, mlp_ratio=4., qkv_bias=True,
                drop=0., attn_drop=0., drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6) if depth else nn.Identity()
        self.rotary = RotaryPosEmbed1D(dim // num_heads, theta, time_scale)
        self.blocks.apply(initialize_transformer)
        self.norm.apply(initialize_transformer)
        # Zero scores start at uniform valid-frame pooling.
        self.score = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, tokens, valid_mask):
        active = valid_mask.to(tokens.device, torch.bool)
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim or active.shape != tokens.shape[:2] or not active.any(1).all():
            raise ValueError("Invalid temporal motion inputs")
        x = tokens.masked_fill(~active[..., None], 0).float()
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite valid motion tokens")
        x = ((x - self.feature_mean) / self.feature_scale).masked_fill(~active[..., None], 0)
        if self.blocks:
            fps = torch.full((len(x),), self.fps, device=x.device)
            rotary = self.rotary(temporal_token_positions(x.shape[1], fps, device=x.device))
            for block in self.blocks:
                x = block(x, active, rotary)
            x = self.norm(x).masked_fill(~active[..., None], 0)
        scores = self.score(x).squeeze(-1).float().masked_fill(~active, -torch.inf)
        weights = scores.softmax(dim=1)
        pooled = (x.float() * weights[..., None]).sum(1)
        return F.normalize(self.linear(pooled).float(), dim=-1)

class TemporalAlignment(nn.Module):
    def __init__(self, text_dim, motion_dim, *, depth=0, num_heads=6, fps=30., theta=100., time_scale=1.):
        super().__init__()
        self.config = AlignmentConfig(text_dim=text_dim, motion_dim=motion_dim,
            embed_dim=motion_dim, text_depth=0, text_pooling="mean", num_heads=1)
        self.text_encoder = _SequenceReadout(text_dim, self.config, depth=0, pooling="mean")
        self.motion_encoder = TemporalReadout(motion_dim, depth, num_heads, fps, theta, time_scale)
    def encode_motion(self, tokens, mask):
        return self.motion_encoder(tokens, mask)
    def encode_text(self, tokens, mask):
        return self.text_encoder(tokens, mask)
    def forward(self, motion, motion_mask, text, text_mask):
        return self.encode_motion(motion, motion_mask), self.encode_text(text, text_mask)

@torch.inference_mode()
def frozen_tokens(encoder, motion, valid_mask, fps, layer, *, bfloat16=False):
    encoder.eval().requires_grad_(False)
    if layer < 1 or layer > len(encoder.blocks) or motion.shape[1] > encoder.num_frames:
        raise ValueError("Invalid layer/frame count")
    extra = encoder.num_frames - motion.shape[1]
    active = F.pad(valid_mask.to(motion.device, torch.bool), (0, extra), value=False)
    if not active.any(1).all():
        raise ValueError("Empty motion")
    cleaned = F.pad(motion.masked_fill(~valid_mask.to(motion.device)[..., None], 0), (0,0,0,extra))
    if not torch.isfinite(cleaned).all():
        raise ValueError("Non-finite valid raw frames")
    captured = []
    hook = encoder.blocks[layer-1].register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with train._autocast(motion.device, bfloat16):
            final = encoder(cleaned, fps, masks=None, valid_frames=active)
            tokens = encoder.norm(captured[0]).masked_fill(~active[..., None], 0)
        if layer == len(encoder.blocks):
            torch.testing.assert_close(tokens, final, rtol=0, atol=0)
        return tokens.float(), active
    finally:
        hook.remove()

class FrameBank:
    def __init__(self, item, records, dim):
        self.item, self.records = item, records
        self.key_to_row = {r["sample_id"]: i for i,r in enumerate(records)}
        self.offsets = np.asarray(item["offsets"], dtype=np.int64)
        self.metadata = {"feature_dim": dim}
        self.values = None
    def __getstate__(self):
        return {**self.__dict__, "values": None}
    def __getitem__(self, key):
        if self.values is None:
            self.values = np.load(self.item["path"], mmap_mode="r", allow_pickle=False)
        row = self.key_to_row[key]
        return torch.from_numpy(np.array(self.values[self.offsets[row]:self.offsets[row+1]], copy=True))

def temporal_datasets(original, manifest, layer):
    dim = manifest["files"]["train"][str(layer)]["shape"][1]
    return {split: PreparedPairDataset(ds.records, ds.text_bank,
        motion_bank=FrameBank(manifest["files"][split][str(layer)], ds.records, dim),
        sample_captions=split=="train") for split,ds in original.items()}

def validate_item(item):
    path = Path(item["path"])
    if _sha256_file(path) != item["sha256"]:
        raise ValueError("Temporal cache checksum differs")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or list(array.shape) != item["shape"]:
        raise ValueError("Temporal cache shape/dtype differs")
    if item["offsets"][0] != 0 or item["offsets"][-1] != len(array) or (np.diff(item["offsets"]) <= 0).any():
        raise ValueError("Invalid frame offsets")
    for start in range(0, len(array), 8192):
        if not np.isfinite(array[start:start+8192]).all():
            raise ValueError("Non-finite temporal cache")

def prepare_tokens(datasets, provenance, encoder, identity, args, device):
    layer = args.layers[0]
    # Validate the existing pooled cache and its exact train-only statistics.
    pooled_args = copy.copy(args)
    pooled_args.feature_cache_root = args.pooled_cache_root
    pooled = probe.prepare_features(datasets, provenance, encoder, identity, pooled_args, torch.device("cpu"))
    signature = dict(source=identity, data=provenance, layer=layer,
        readout="shared_frozen_terminal_LN_per_frame_FP32", precision="BF16" if args.use_bfloat16 else "FP32",
        batch_size=args.eval_batch_size, pooled_manifest_sha256=json_digest(pooled),
        normalization="original_train_pooled_mean_scale_applied_per_token",
        split_masks={k: json_digest([[r["sample_id"], r["length"], r["fps"]] for r in ds.records]) for k,ds in datasets.items()})
    root = Path(args.feature_cache_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root/"manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Temporal cache recipe differs")
        for items in manifest["files"].values():
            validate_item(items[str(layer)])
        return manifest
    required = sum(sum(r["length"] for r in ds.records) for ds in datasets.values()) * encoder.embed_dim * 4
    existing = sum(p.stat().st_size for p in root.glob("*.npy"))
    if shutil.disk_usage(root).free < max(0, required-existing) + 2*1024**3:
        raise RuntimeError("Insufficient local cache disk space")
    encoder.to(device).eval().requires_grad_(False)
    # Validate prefix against the full encoder on a real batch before truncating.
    sample = collate_pairs([datasets["train"][i] for i in range(min(2,len(datasets["train"])))])
    motion, mask = sample["motion_tokens"].to(device), sample["motion_mask"].to(device)
    fps = torch.full((len(motion),), provenance["provenance"]["fps"], device=device)
    full, active = frozen_tokens(encoder, motion, mask, fps, layer, bfloat16=args.use_bfloat16)
    encoder.blocks = nn.ModuleList(list(encoder.blocks)[:layer])
    prefix, _ = frozen_tokens(encoder, motion, mask, fps, layer, bfloat16=args.use_bfloat16)
    torch.testing.assert_close(prefix, full, rtol=0, atol=0)
    files = {}
    for split, source in datasets.items():
        done = root/(split+"-complete.json")
        if done.exists():
            item = json.loads(done.read_text())
            if item["signature_sha256"] != json_digest(signature):
                raise ValueError("Completed split recipe differs")
            validate_item(item)
            files[split] = {str(layer): item}
            continue
        ds = copy.copy(source); ds.sample_captions = False
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=device.type=="cuda")
        offsets = np.concatenate(([0], np.cumsum([r["length"] for r in ds.records], dtype=np.int64)))
        temporary = root/(split+"-frames.tmp.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
            shape=(int(offsets[-1]), encoder.embed_dim))
        old_vectors = np.load(pooled["files"][split][str(layer)]["path"], mmap_mode="r")
        cursor, max_error = 0, 0.
        for batch in loader:
            motion, active = batch["motion_tokens"].to(device), batch["motion_mask"].to(device)
            fps = torch.full((len(motion),), provenance["provenance"]["fps"], device=device)
            tokens, valid = frozen_tokens(encoder, motion, active, fps, layer, bfloat16=args.use_bfloat16)
            lengths = valid.sum(1)
            if not torch.equal(valid, torch.arange(valid.shape[1], device=device)[None,:] < lengths[:,None]):
                raise ValueError("Ragged cache requires contiguous valid frame prefixes")
            pooled_now = (tokens.sum(1)/lengths[:,None]).cpu().numpy()
            # Same extraction batch size, weights and precision: require exact cached mean parity.
            np.testing.assert_array_equal(pooled_now, old_vectors[cursor:cursor+len(tokens)])
            cpu = tokens.cpu().numpy()
            for row, length in enumerate(lengths.tolist()):
                if length != ds.records[cursor]["length"]:
                    raise ValueError("Dataset mask/record length differs")
                values[offsets[cursor]:offsets[cursor+1]] = cpu[row,:length]
                cursor += 1
            if cursor % (args.eval_batch_size*20) == 0:
                values.flush()
                print(f"Frame extraction {split}: {cursor}/{len(ds)}", flush=True)
        if cursor != len(ds):
            raise ValueError("Incomplete frame extraction")
        values.flush(); del values
        path = root/(split+"-frames.npy"); os.replace(temporary,path)
        item = dict(path=str(path), shape=[int(offsets[-1]), encoder.embed_dim],
            sha256=_sha256_file(path), offsets=offsets.tolist(), sample_count=len(ds),
            signature_sha256=json_digest(signature), pooled_parity="exact_all_rows")
        _atomic_json_save(item, done)
        files[split] = {str(layer): item}
        print(f"Frame extraction complete: {split}", flush=True)
    encoder.to("cpu")
    if device.type=="cuda": torch.cuda.empty_cache()
    manifest = dict(signature=signature, files=files, statistics=pooled["statistics"],
        prefix_hook_parity=True, encoder_frozen=True, format_version=1)
    _atomic_json_save(manifest, marker)
    return manifest

def model_config(args, manifest):
    identity = manifest["signature"]["source"]
    return dict(text_dim=manifest["signature"]["data"]["text_dim"],
        motion_dim=manifest["files"]["train"][str(args.layers[0])]["shape"][1],
        layer=args.layers[0], motion_readout="attention_pool_linear_L2",
        text_readout="linear_LN_mean_L2", readout_depth=args.readout_depth,
        num_heads=identity["encoder"]["num_heads"] if "encoder" in identity else 6,
        fps=manifest["signature"]["data"]["provenance"]["fps"], theta=100., time_scale=1.,
        normalization="original_train_pooled_mean_scale_per_token",
        mlp_ratio=4., qkv_bias=True, norm_eps=1e-6, dropout=0., attention_initialization="uniform")

def make_model(config):
    return TemporalAlignment(config["text_dim"],config["motion_dim"],
        depth=config["readout_depth"],num_heads=config["num_heads"],fps=config["fps"],
        theta=config["theta"],time_scale=config["time_scale"])

def fit(datasets, manifest, args, device):
    config = model_config(args, manifest)
    return probe.fit_probe(temporal_datasets(datasets, manifest, args.layers[0]), manifest,
        args.layers[0], args, device, model_factory=lambda text,dim: make_model(config),
        model_config=config, experiment=EXPERIMENT)

def run(args):
    if args.input_source != "raw" or args.layers not in ([6], [12]) or args.model_size != "base":
        raise ValueError("This controlled experiment requires raw input, base, layer 6 or12")
    if not args.validation_only:
        raise ValueError("Use validation-only training; select before final test")
    check=copy.copy(args); check.motion_adaptation="lora"; check.model_version="v2"
    train._validate_args(check); probe.probe_optimization(args)
    device=resolve_device(args.device)
    datasets, provenance=load_prepared_datasets(**train.data_configuration(args))
    encoder, identity=load_motion_initialization(args.jepa_checkpoint,args.checkpoint_key,provenance,args.model_size)
    if identity["checkpoint_sha256"] != args.expected_source_sha256:
        raise ValueError("Unexpected original encoder checkpoint")
    if identity["pretrain_epoch"] != 300 or identity["checkpoint_key"] != "target_encoder":
        raise ValueError("Require original final EMA encoder")
    manifest=prepare_tokens(datasets,provenance,encoder,identity,args,device)
    del encoder
    if args.extract_only:
        return {"cache_complete": True, "manifest": str(Path(args.feature_cache_root)/"manifest.json")}
    return fit(datasets,manifest,args,device)

def build_parser():
    p=probe.build_parser()
    p.add_argument("--pooled-cache-root",type=Path,required=True)
    p.add_argument("--readout-depth",type=int,choices=[0,1,2],default=0)
    p.add_argument("--expected-source-sha256",required=True)
    p.add_argument("--extract-only",action="store_true")
    return p

if __name__=="__main__":
    print(json.dumps(run(build_parser().parse_args()),indent=2),flush=True)
