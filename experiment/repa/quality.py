"""Periodic Kimodo TMR-FID and foot-skating evaluation of unconditional samples."""
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from experiment.utils import _atomic_json_save, _sha256_file
from motion_rep.registry import get_representation
from skeleton.definitions import SOMASkeleton30
from .objectives import sample_motion

UPSTREAM_COMMIT = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
MODEL_REPO = "nvidia/TMR-SOMA-RP-v1"
MODEL_REVISION = "e427752ae3446dedba49e928c93ddc9f0e413401"


def evaluation_settings(config):
    settings = dict(config.get("generation_evaluation", {}))
    if not settings.get("enabled", False):
        return None
    defaults = dict(interval=10000, samples=1024, batch_size=32, sampling_steps=[128],
                    seed=42, checkpoint_dir=None, on_completion=False)
    defaults.update(settings)
    for key in ("interval", "samples", "batch_size"):
        if isinstance(defaults[key], bool) or not isinstance(defaults[key], int) or defaults[key] < 1:
            raise ValueError(f"generation_evaluation.{key} must be a positive integer")
    if defaults["samples"] < 2:
        raise ValueError("FID requires at least two samples")
    steps = defaults["sampling_steps"]
    base = config["shortcut"]["base_steps"]
    if not isinstance(steps, list) or not steps or len(set(steps)) != len(steps):
        raise ValueError("sampling_steps must be a nonempty list of distinct powers of two")
    for value in steps:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > base or value & (value-1):
            raise ValueError("Invalid evaluation sampling step count")
    if config["data"]["fps"] != 30:
        raise ValueError("The pretrained Kimodo TMR evaluator requires 30 FPS data")
    return defaults


def quality_due(settings, step, final=False):
    return settings is not None and (step % settings["interval"] == 0 or
                                    (final and settings["on_completion"]))


@contextlib.contextmanager
def evaluation_context(device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device.type, enabled=False):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


class MotionEmbeddingModel:
    """Only the released TMR motion encoder; never loads an LLM/text encoder."""
    def __init__(self, checkpoint_dir, device):
        from .kimodo_eval.encoder import ACTORStyleEncoder
        from .kimodo_eval.representation import TMRMotionRep
        downloaded = checkpoint_dir is None
        if checkpoint_dir is None:
            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                MODEL_REPO, revision=MODEL_REVISION,
                allow_patterns=["config.yaml", "last_weights/motion_encoder.pt",
                                "stats/motion/*", "LICENSE"])
        root = Path(checkpoint_dir)
        config = yaml.safe_load((root / "config.yaml").read_text())
        rep = config["motion_rep"]
        if rep["fps"] != 30 or rep["skeleton"]["_target_"].split(".")[-1] != "SOMASkeleton30":
            raise ValueError("Expected the Kimodo SOMA30, 30 FPS TMR evaluator")
        if rep["_target_"].split(".")[-1] != "TMRMotionRep":
            raise ValueError("Unsupported TMR motion representation")
        self.skeleton = SOMASkeleton30().to(device)
        self.rep = TMRMotionRep(self.skeleton, root / "stats/motion", fps=30).to(device)
        kwargs = {key: config[key] for key in ("vae", "latent_dim", "ff_size", "num_layers", "num_heads", "dropout", "activation")}
        self.encoder = ACTORStyleEncoder(motion_rep=self.rep, llm_shape=None, **kwargs).to(device)
        weights = root / "last_weights/motion_encoder.pt"
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state.get("state_dict", state), strict=True)
        self.encoder.requires_grad_(False).eval()
        self.identity = {
            "repo": MODEL_REPO, "revision": MODEL_REVISION if downloaded else None,
            "motion_encoder_sha256": _sha256_file(weights),
            "config_sha256": _sha256_file(root / "config.yaml"),
            "stats_sha256": {str(path.relative_to(root)): _sha256_file(path) for path in self.rep.stats_files},
            "unit_vector": True, "sample_mean": True, "upstream_commit": UPSTREAM_COMMIT,
        }

    def __call__(self, joints, lengths):
        if joints.shape[2:] != (30,3) or (lengths < 2).any():
            raise ValueError("TMR requires SOMA30 motions with at least two valid frames")
        valid = torch.arange(joints.shape[1], device=joints.device)[None] < lengths[:,None]
        features = self.rep(posed_joints=joints.float(), lengths=lengths,
                            to_canonicalize=True, to_normalize=True)
        features = features.masked_fill(~valid[...,None], 0)
        tokens = self.encoder({"x": features, "mask": valid})
        return torch.nn.functional.normalize(tokens[:,0].float(), dim=-1)


class GenerationEvaluator:
    def __init__(self, config, ds, mean, std, device):
        from .kimodo_eval.foot_skate import FootSkateFromHeight, FootSkateFromContacts, FootSkateRatio
        self.settings = evaluation_settings(config)
        self.config, self.ds, self.device = config, ds, device
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)
        count = min(self.settings["samples"], len(ds))
        if count < 2:
            raise ValueError("Validation dataset must contain at least two motions for FID")
        self.indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(self.settings["seed"]))[:count].tolist()
        with evaluation_context(device):
            self.embedding = MotionEmbeddingModel(self.settings["checkpoint_dir"], device)
        self.rep = get_representation(config["data"]["representation"])(self.embedding.skeleton, 30)
        self.metrics = [FootSkateFromHeight(skeleton=self.embedding.skeleton, fps=30),
                        FootSkateFromContacts(skeleton=self.embedding.skeleton, fps=30),
                        FootSkateRatio(skeleton=self.embedding.skeleton, fps=30)]
        self.reference = None

    def decode(self, motion):
        output = self.rep.inverse(motion.float() * self.std + self.mean)
        if not torch.isfinite(output["posed_joints"]).all():
            raise FloatingPointError("Non-finite decoded motion")
        return output

    def foot_metrics(self, output, lengths):
        result = {}
        for metric in self.metrics:
            result.update(metric._compute(posed_joints=output["posed_joints"],
                           foot_contacts=output["foot_contacts"], lengths=lengths))
        return result

    @staticmethod
    def summarize(values):
        return {key: float(torch.cat(parts).double().mean()) for key, parts in values.items()}

    def run(self, ema, step):
        was_training = ema.training
        ema.eval()
        try:
            with evaluation_context(self.device):
                return self._run(ema, step)
        finally:
            ema.train(was_training)

    def _run(self, ema, step):
        from .kimodo_eval.fid import calculate_activation_statistics, calculate_frechet_distance
        cfg = self.settings
        reference_features, reference_feet = [], {}
        generated = {steps: {"features": [], "feet": {}} for steps in cfg["sampling_steps"]}
        for start in range(0, len(self.indices), cfg["batch_size"]):
            indices = self.indices[start:start+cfg["batch_size"]]
            clean, fps, lengths = [x.to(self.device) for x in torch.utils.data.default_collate([self.ds[i] for i in indices])]
            if (lengths < 2).any() or not (fps == 30).all():
                raise ValueError("Quality evaluation requires >=2 frames and 30 FPS")
            valid = torch.arange(clean.shape[1], device=self.device)[None] < lengths[:,None]
            if self.reference is None:
                real = self.decode(clean)
                reference_features.append(self.embedding(real["posed_joints"], lengths).cpu().numpy())
                for key, value in self.foot_metrics(real, lengths).items():
                    reference_feet.setdefault(key, []).append(value.cpu())
            # Per-sample seeds keep comparisons reproducible across evaluation batch sizes.
            noise = torch.stack([torch.randn(clean.shape[1:], device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(cfg["seed"] + index + 100000))
                for index in indices])
            for steps in cfg["sampling_steps"]:
                motion = sample_motion(ema, noise, fps.float(), valid, steps, self.config["shortcut"]["base_steps"])
                output = self.decode(motion)
                generated[steps]["features"].append(self.embedding(output["posed_joints"], lengths).cpu().numpy())
                for key, value in self.foot_metrics(output, lengths).items():
                    generated[steps]["feet"].setdefault(key, []).append(value.cpu())
        if self.reference is None:
            features = np.concatenate(reference_features).astype(np.float64)
            self.reference = (calculate_activation_statistics(features), self.summarize(reference_feet))
        reference_stats, real_metrics = self.reference
        result = {"step": step, "num_samples": len(self.indices), "seed": cfg["seed"],
                  "dataset_root": str(getattr(self.ds, "root_path", "")), "indices": self.indices,
                  "representation": self.config["data"]["representation"],
                  "embedding_model": self.embedding.identity, "reference": real_metrics, "sampling": {}}
        for steps, values in generated.items():
            features = np.concatenate(values["features"]).astype(np.float64)
            if not np.isfinite(features).all():
                raise FloatingPointError("Non-finite TMR embeddings")
            fid = float(calculate_frechet_distance(*calculate_activation_statistics(features), *reference_stats))
            if not math.isfinite(fid) or fid < -1e-5:
                raise FloatingPointError(f"Invalid FID: {fid}")
            metrics = dict(fid=max(0.,fid), **self.summarize(values["feet"]))
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Non-finite quality metric")
            result["sampling"][str(steps)] = metrics
        return result


def distributed_quality(evaluator, ema, step, output, writer, rank, world):
    """Rank zero evaluates the full distribution; propagate errors to every rank."""
    message = [None]
    if rank == 0:
        try:
            result = evaluator.run(ema, step)
            _atomic_json_save(result, Path(output) / "quality" / f"step-{step:08d}.json")
            for steps, metrics in result["sampling"].items():
                for name, value in metrics.items():
                    if writer:
                        writer.add_scalar(f"quality/steps_{steps}/{name}", value, step)
            if writer:
                for name,value in result["reference"].items():
                    writer.add_scalar("quality/reference/"+name, value, step)
                writer.flush()
            print(json.dumps({"step":step,"quality":result["sampling"]}),flush=True)
            message[0] = {"ok":True}
        except Exception as error:
            message[0] = {"error":f"{type(error).__name__}: {error}"}
    if world > 1:
        dist.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        raise RuntimeError("Generation evaluation failed: "+message[0]["error"])


def initialize_evaluator(config, ds, mean, std, device, rank, world, output):
    settings = evaluation_settings(config)
    if settings is None:
        return None
    evaluator = None
    message = [None]
    if rank == 0:
        try:
            evaluator = GenerationEvaluator(config, ds, mean, std, device)
            (Path(output)/"quality").mkdir(parents=True,exist_ok=True)
            message[0] = {"ok":True}
        except Exception as error:
            message[0] = {"error":f"{type(error).__name__}: {error}"}
    if world > 1:
        dist.broadcast_object_list(message,src=0)
    if "error" in message[0]:
        raise RuntimeError("Could not initialize generation evaluator: "+message[0]["error"])
    return evaluator
