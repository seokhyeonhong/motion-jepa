# Dataset Visualization Summary

## Purpose

The visualization tool is a standalone browser for processed Motion-JEPA NPY
motions. It uses `viser` for the web interface and scene, the local
Motion-JEPA decoder for reconstruction, and local SOMA assets for skeleton and
mesh rendering. It does not depend on Kimodo visualization code.

The entry point is:

```bash
python visualize_dataset.py
```

For a path-oriented browser whose dropdown selects an NPY file, use:

```bash
python dataset/visualize.py
```

It lists paths from the selected split (500 by default) and provides playback,
frame stepping, speed, stored FPS, skinned-mesh visibility and opacity,
skeleton visibility, and foot-contact highlighting. Pass `--limit 0` to list
every processed path or `--mesh` to start with the skinned body visible.

The default input is `dataset/bones-seed-processed` and the default server is
`0.0.0.0:6006`.

## Usage

Typical commands:

```bash
# Browse the first 500 training motions as skeletons.
python visualize_dataset.py

# Start with the skinned SOMA mesh visible.
python visualize_dataset.py dataset/bones-seed-processed --split train --mesh

# Open a particular item from a split.
python visualize_dataset.py --split val --sample_index 10
python visualize_dataset.py --split test --sample_id <motion-id>
```

Available arguments:

| Argument | Meaning |
|---|---|
| `input` | Motion-JEPA NPY dataset directory |
| `--split` | `train`, `val`, or `test` |
| `--sample_index` | Initially selected entry index |
| `--sample_id` | Initially selected exact motion ID |
| `--limit` | Maximum discovered entries; default 500, non-positive means unlimited |
| `--host` | Server address; default `SERVER_NAME` or `0.0.0.0` |
| `--port` | Server port; default `SERVER_PORT` or 6006 |
| `--mesh` | Show the mesh initially instead of the skeleton |
| `--normalized` | Treat stored features as normalized and undo normalization |

After launch, open the printed HTTP address in a browser. If the server runs
remotely, use the reachable host address or an SSH port forward.

## Dataset Discovery

The viewer reads `<split>.txt` in split order and uses `index.json` to attach
the first available caption. If the split file is unavailable, it can discover
NPY paths directly from `index.json`.

Long source motions appear as independently canonicalized segment IDs ending
in `_0000`, `_0001`, and so on. An exact one-window source keeps its original
ID. Default segments contain 300 frames at 60 FPS (five seconds).

The viewer decodes raw motion values from the selected NPY file. FPS is read from the
processed dataset's `meta.json`. With `--normalized`, it loads
`stats/mean.npy` and `stats/std.npy` and reverses normalization before decoding.

## Decoding and Rendering

The stored 366-dimensional features are decoded with
`MotionJEPAMotionRep(SOMASkeleton30(), fps)`. The decoded SOMA30 motion is then
expanded to SOMA77 for display.

Two renderers are available:

- A shaded articulated skeleton with joint spheres and bone capsules.
- A skinned SOMA body mesh using local linear-blend skinning assets.

Mesh vertices are computed only for the currently displayed frame rather than
caching every skinned frame. This limits memory use for long motions. The
ground is an infinite XZ grid and the scene uses Y as the up direction.

Foot contacts stored in the representation can be displayed on the six
expanded foot joints. Contacted joints are highlighted by the skeleton
renderer.

## Browser Controls

The GUI provides:

- A motion selector and motion ID/path/caption display.
- Play/pause.
- A frame slider and previous/next frame buttons.
- Playback speeds of 0.5x, 1x, and 2x.
- Read-only effective FPS.
- Independent mesh and skeleton visibility toggles.
- Mesh opacity.
- Foot-contact highlighting.

Playback uses each segment's stored configured FPS. Reaching the final frame
loops back to frame zero.

## Session Model

Each connected browser client receives its own `ViewerSession`, selection,
renderer, playback state, speed, and GUI controls. Changing motion clears the
old scene handles, loads and decodes the selected NPY file, resets playback to
frame zero, and updates the displayed FPS and metadata.

A background loop runs at up to 240 checks per second and advances active
sessions according to `1 / (fps * speed)`. Disconnecting a client removes its
session, while stopping the command with Ctrl-C closes the viewer loop.

## Dependencies and Implementation

Install visualization dependencies with:

```bash
pip install -r requirements-motion.txt
```

The primary implementation files are:

```text
visualize_dataset.py
visualization/dataset_viewer.py
visualization/shaded_skeleton.py
visualization/soma_skin.py
```

If `viser` is unavailable, startup reports that `viser` and `trimesh` must be
installed.
