# Upstream provenance

Copied/adapted from https://github.com/nv-tlabs/kimodo at commit
e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 (Apache-2.0; see LICENSE).

- encoder.py: PositionalEncoding and ACTORStyleEncoder from kimodo/model/tmr.py.
- features.py: velocity, heading, rotation helpers from motion_rep/feature_utils.py,
  plus contact detection from motion_rep/feet.py.
- representation.py: TMRMotionRep feature construction, rotation and translation.
- foot_skate.py: height/contact/ratio metrics from metrics/foot_skate.py.
- fid.py: activation statistics and Frechet distance from metrics/retrieval.py.
- base.py: metrics/base.py.

Adaptations remove unbatched decorators, text-model dependencies, generation-only
representation methods and Hydra loading. The local wrapper supplies batched
SOMA30 FK positions, canonicalizes features, applies sqrt(std**2+1e-5)
normalization, and uses the deterministic first VAE token (mu), L2-normalized,
matching benchmark/embed_folder.py. Metrics retain upstream formulas and thresholds.
TMR weights are downloaded separately under their own model license; they are not
redistributed here.
