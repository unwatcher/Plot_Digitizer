# Contributing to PlotDigitizer

Help make scientific figures easier to turn into reviewed numerical data.
Useful contributions include documentation fixes, reproducible extraction
reports, spectrum workflow examples and focused code changes.

## Find a starting point

Check [open issues](https://github.com/unwatcher/Plot_Digitizer/issues) and
[good first issues](https://github.com/unwatcher/Plot_Digitizer/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22).
Small documentation corrections can go straight to a pull request. For a new
algorithm, dependency, export format or substantial UI change, open an issue
first to discuss the expected behavior and scope.

A helpful first contribution can be a clearer setup step, a small synthetic
plot that reproduces a bug, or a documented PXRD/Raman/IR export workflow.

## Report a problem or propose a feature

Use the [issue forms](https://github.com/unwatcher/Plot_Digitizer/issues/new/choose).
Include the release tag or commit, operating system, launch method and exact
steps. For extraction or calibration problems, include the expected curve or
axis values and the ROI, preset or threshold you used, when relevant.

Prefer a bundled example or a minimal synthetic image. Attach only images
you can share publicly; remove private laboratory information. For an external
figure, include its source and reuse basis. See the
[example provenance note](examples/README.md).

## Set up the source

Use 64-bit CPython 3.14. From a terminal:

```text
git clone https://github.com/unwatcher/Plot_Digitizer.git
cd Plot_Digitizer
python -m venv .venv
```

Activate `.venv` using the command for your shell:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Linux (Bash) | `source .venv/bin/activate` |

Then install dependencies and launch:

```text
python -m pip install -r requirements.txt
python plot_digitizer_auto.py
```

For headless extraction and API work, install `requirements-core.txt` instead
of the desktop requirements. Start the API with `python -m plot_digitizer.api`;
see the [API guide](docs/API.md). Automatic calibration uses installed Windows
OCR language support, with Tesseract as a fallback. Linux needs Tesseract
for automatic OCR. Manual calibration works without it.

## Verify a change

Run the checks for the affected behavior. The full desktop suite is:

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

For unattended desktop tests, set `QT_QPA_PLATFORM=offscreen` in the environment.
For a core-only installation, run these suites individually:

```text
python -m unittest discover -s tests -p test_digitizer_core.py -v
python -m unittest discover -s tests -p test_axis_ocr.py -v
python -m unittest discover -s tests -p test_patent_plot_corpus.py -v
python -m unittest discover -s tests -p test_api.py -v
```

For extraction changes, also run `python scripts/benchmark_plot_digitizer.py`
and inspect the overlays under `output/patent_digitizer_benchmark/current`.
Use `--calibration` when checking the installed OCR engine. The bundled
examples are regression cases, not an independent scientific accuracy study.

For documentation-only changes, check links, commands and rendered Markdown.
For UI changes, include a screenshot and the manual workflow you exercised.
For release packaging changes, follow the
[Windows build and archive verification steps](README.md#building-windows-packages).
CI checks the headless core on Ubuntu and the desktop on Windows and Ubuntu.

## Keep scientific outputs reviewable

- Keep source-image pixels distinct from calibrated scientific values.
- Preserve observed samples and floating-point CSV/JSON values. Explain any
  new smoothing, interpolation, resampling or intensity conversion explicitly.
- Keep headless core/API modules independent of Qt and Matplotlib.
- Cover a behavior change with a focused regression check where appropriate.
- Document limitations and example provenance; avoid unsupported accuracy claims.
- Keep changes focused and retain license and third-party notices.

Open a pull request against `main` explaining the user-visible change, how you
verified it, and any remaining limitation. Use the supplied PR template.

## Documentation images

Recreate the scientific before/after example with
`python scripts/render_example.py`. The social card is generated separately
from the bundled figure and CSV; see [social preview assets](docs/social-preview.md).

Contributions are distributed under the repository's
[BSD 3-Clause license](LICENSE). Source figures retain their separate reuse
terms. Keep discussion respectful and focused on reproducible behavior.
