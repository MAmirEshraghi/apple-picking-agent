"""
Stable-Baselines3 algorithms and helpers for apple picking: PPO+BC with LSTM policy (algo/PPOLSTMAE),
image encoder, optical flow, running statistics and training callbacks (apple_gym/). Adapted from an
OSU Robotics SB3 codebase.

Checkpoints pickle classes by module path. Names used by earlier versions of this package are mapped to
the current modules below, so older checkpoints still load with RecurrentPPOAEWithExpert.load.
"""
import importlib
import importlib.abc
import importlib.util
import sys

# Earlier module prefix -> current module prefix (longest match wins).
_LEGACY_PREFIXES = {
    "apple_picking_sb3.pruning_gym": __name__ + ".apple_gym",
    "apple_picking_sb3": __name__,
    __name__ + ".pruning_gym": __name__ + ".apple_gym",
}


def _current_name(fullname):
    for old in sorted(_LEGACY_PREFIXES, key=len, reverse=True):
        if fullname == old or fullname.startswith(old + "."):
            return _LEGACY_PREFIXES[old] + fullname[len(old):]
    return None


class _LegacyNameAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        return importlib.util.spec_from_loader(fullname, self) if _current_name(fullname) else None

    def create_module(self, spec):
        return importlib.import_module(_current_name(spec.name))

    def exec_module(self, module):
        pass


if not any(isinstance(finder, _LegacyNameAlias) for finder in sys.meta_path):
    # First in the chain: otherwise the path finder loads a second copy of child modules from disk.
    sys.meta_path.insert(0, _LegacyNameAlias())
