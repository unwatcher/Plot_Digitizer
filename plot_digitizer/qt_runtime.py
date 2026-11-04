"""Small, import-safe Qt runtime environment fixes for offline Qt use."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path


HEADLESS_QT_PLATFORMS = {"offscreen", "minimal", "minimalegl"}


def _application_root() -> Path:
    """Return the source/release root for either supported layout.

    In CrystalTrace this module lives in ``crystaltrace/``.  The standalone
    PlotDigitizer archive copies it next to ``plot_digitizer.py``.
    """
    module_parent = Path(__file__).resolve().parent
    if module_parent.name.casefold() in {"crystaltrace", "plot_digitizer"}:
        return module_parent.parent
    return module_parent


def _release_candidates(application_root: Path) -> tuple[Path, ...]:
    runtime_root = application_root / "runtime_deps" / "py314"
    return (
        runtime_root / "matplotlib" / "mpl-data" / "fonts" / "ttf",
        application_root / "matplotlib" / "mpl-data" / "fonts" / "ttf",
        application_root / "qt_fonts",
    )


def configure_qt_font_directory(
    environ: MutableMapping[str, str] | None = None,
    *,
    platform: str | None = None,
    candidates: Sequence[str | Path] | None = None,
    application_root: str | Path | None = None,
) -> str:
    """Point headless Qt at installed system fonts before importing PySide6.

    The Windows/offscreen Qt plugin cannot use the normal Windows font database
    and otherwise may probe an absent bundled font directory.
    Visible Windows launches continue to use the native platform font database.
    """
    environ = os.environ if environ is None else environ
    qpa_platform = str(environ.get("QT_QPA_PLATFORM", "")).split(":", 1)[0]
    if qpa_platform.strip().casefold() not in HEADLESS_QT_PLATFORMS:
        return ""
    configured = str(environ.get("QT_QPA_FONTDIR", "")).strip()
    if configured and Path(configured).is_dir():
        return configured
    current_platform = sys.platform if platform is None else platform
    if candidates is None:
        candidates = ()
        if current_platform.casefold().startswith("win"):
            windows_root = str(
                environ.get("WINDIR") or environ.get("SystemRoot") or r"C:\Windows"
            ).strip()
            candidates = (Path(windows_root) / "Fonts",)
        if application_root is None:
            application_root = _application_root()
        # Prefer the normal Windows font database, then fall back to fonts
        # shipped inside the release. The latter is important for a copied
        # offline folder and headless/offscreen Qt, where Qt otherwise probes
        # an absent bundled Qt font directory.
        candidates = (*candidates, *_release_candidates(Path(application_root)))
    for candidate in candidates:
        font_directory = Path(candidate)
        if font_directory.is_dir():
            resolved = str(font_directory.resolve())
            environ["QT_QPA_FONTDIR"] = resolved
            return resolved
    return ""


def configure_qt_runtime(
    environ: MutableMapping[str, str] | None = None,
    *,
    application_root: str | Path | None = None,
) -> str:
    """Configure bundled Qt plugin and font paths before importing PySide6.

    PySide6 normally discovers these paths from an installed wheel. An offline
    release is an extracted directory, so explicitly pointing Qt at its local
    plugin tree makes launches independent of the current working directory.
    """
    environ = os.environ if environ is None else environ
    root = Path(application_root) if application_root is not None else _application_root()
    runtime_root = root / "runtime_deps" / "py314"
    plugin_directory = runtime_root / "PySide6" / "plugins"
    environ["QT_API"] = "pyside6"
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        if "pyqt5" in str(environ.get(key, "")).casefold():
            environ.pop(key, None)
    native_plugin = {"win32": "qwindows.dll", "darwin": "libqcocoa.dylib"}.get(sys.platform, "libqxcb.so")
    if (plugin_directory / "platforms" / native_plugin).is_file():
        configured_plugins = str(environ.get("QT_PLUGIN_PATH", "")).strip()
        if not configured_plugins or not Path(configured_plugins).is_dir():
            environ["QT_PLUGIN_PATH"] = str(plugin_directory.resolve())
        platform_plugins = plugin_directory / "platforms"
        configured_platform_plugins = str(
            environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
        ).strip()
        if platform_plugins.is_dir() and (
            not configured_platform_plugins
            or not Path(configured_platform_plugins).is_dir()
        ):
            environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins.resolve())
    return configure_qt_font_directory(environ, application_root=root)
