"""Register ``modules`` → ``zepto.modules`` when not installed as top-level."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys


class _ModulesAliasFinder:
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != "modules" and not fullname.startswith("modules."):
            return None
        target_name = "zepto." + fullname
        spec = importlib.util.find_spec(target_name)
        if spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _ModulesAliasLoader(target_name),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )


class _ModulesAliasLoader(importlib.abc.Loader):
    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:
        return importlib.import_module(self._target_name)

    def exec_module(self, module: object) -> None:
        return None


def ensure_import_paths() -> None:
    if "modules" in sys.modules:
        return
    try:
        importlib.import_module("modules")
        return
    except ModuleNotFoundError:
        pass
    for finder in sys.meta_path:
        if isinstance(finder, _ModulesAliasFinder):
            return
    sys.meta_path.insert(0, _ModulesAliasFinder())
