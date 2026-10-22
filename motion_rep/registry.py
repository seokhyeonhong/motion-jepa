"""Supported preprocessing and training representations."""
from .reps import LocalMotionJEPARep, MotionJEPAMotionRep

REPRESENTATIONS = {
    "MotionJEPAMotionRep": MotionJEPAMotionRep,
    "LocalMotionJEPARep": LocalMotionJEPARep,
}
DEFAULT_REPRESENTATION = "MotionJEPAMotionRep"


def get_representation(name=DEFAULT_REPRESENTATION):
    for key, cls in REPRESENTATIONS.items():
        if name in (key, cls.FORMAT_NAME):
            return cls
    raise ValueError(f"Unknown motion representation {name!r}; choose {list(REPRESENTATIONS)}")


def add_representation_argument(parser):
    parser.add_argument("--representation", choices=list(REPRESENTATIONS),
                        default=DEFAULT_REPRESENTATION,
                        help="Motion feature representation (use a separate output directory).")


def validate_representation_output(args):
    """Never silently reuse a dataset encoded with a different representation."""
    import json
    from pathlib import Path
    cls = get_representation(getattr(args, "representation", DEFAULT_REPRESENTATION))
    metadata_path = Path(args.output) / "meta.json"
    if metadata_path.is_file() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("representation") != cls.FORMAT_NAME:
            raise ValueError("Existing dataset uses another representation; use a new --output "
                             "directory or explicit --overwrite")
    return cls
