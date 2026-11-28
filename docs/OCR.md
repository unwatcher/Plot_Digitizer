# Local OCR for automatic calibration

PlotDigitizer reads axis labels with a local OCR engine. Install Tesseract once
on macOS or Ubuntu. After installation, opening a figure and reading its
labels works offline.

## Install Tesseract

On macOS, use the [Homebrew formula](https://formulae.brew.sh/formula/tesseract).

```sh
brew install tesseract
```

On Ubuntu, install the engine and English language data. See the
[Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html#ubuntu).

```sh
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

From the terminal used to launch PlotDigitizer, check the executable and its language data.

```sh
command -v tesseract
tesseract --version
tesseract --list-langs
```

If the first command prints no path, add Tesseract's `bin` directory to `PATH`
and open a new terminal. On macOS, make sure the terminal loads Homebrew's shell
environment. Launch PlotDigitizer from that terminal so it sees the same `PATH`.
If the executable is found but `eng` is missing, reinstall the package that provides
the English language data.

## Try a bundled plot

Open [`us8198452b2-fig1.png`](../examples/patent_plots/us8198452b2-fig1.png)
and run **Auto-calibration**. Review the proposed tick positions and values
against the figure. If OCR misses a label or reads it incorrectly, edit the tick
row or click the correct tick and enter its value. Use manual calibration if the
proposals are incomplete or if Tesseract is unavailable.

For a headless check with the source checkout and `requirements-core.txt` installed,
run the examples benchmark with local OCR enabled.

```sh
python scripts/benchmark_plot_digitizer.py --split examples --calibration
```

Inspect the generated report and overlays. The benchmark records proposals for
review. Check the axis values against the source figure before applying them.
