"""YAML configuration: packaged defaults deep-merged with user overrides."""

from pathlib import Path

from ruamel.yaml import YAML

DEFAULTS_PATH = Path(__file__).parent / "config.defaults.yaml"
USER_CONFIG_PATH = Path("~/.config/localflow/config.yaml").expanduser()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(user_path: str | Path | None = None) -> dict:
    """Load packaged defaults, then overlay the user's config file if present."""
    yaml = YAML(typ="safe")
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.load(f)
    path = Path(user_path).expanduser() if user_path else USER_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            user_cfg = yaml.load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg
