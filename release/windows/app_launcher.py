from __future__ import annotations

import ctypes
import datetime
import os
import sys
import tempfile
import traceback
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "plot_digitizer_launch.log"
DEPS = APP_DIR / "runtime_deps" / "py314"

paths = (APP_DIR, DEPS) if sys.platform == "win32" else (APP_DIR,)
for path in paths:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
from plot_digitizer.qt_runtime import configure_qt_runtime

configure_qt_runtime(application_root=APP_DIR)


def log(message: str) -> None:
    global LOG_PATH
    entry = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}\n"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError:
        LOG_PATH = Path(tempfile.gettempdir()) / "plotdigitizer_launch.log"
        try:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(entry)
        except OSError:
            print(entry, file=sys.stderr)


def show_error(message: str) -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, message, "PlotDigitizer - Startup Error", 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> int:
    log("Starting PlotDigitizer")
    try:
        from plot_digitizer.gui import main as run

        run(standalone=True)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 0
        log(f"Application exited with code {code}")
        return code
    except BaseException:
        details = traceback.format_exc()
        log(details)
        show_error(
            "The application could not start.\n\n"
            f"Details were written to:\n{LOG_PATH}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
