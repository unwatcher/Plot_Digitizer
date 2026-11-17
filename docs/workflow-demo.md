# Image-to-export walkthrough

These static screenshots show the Windows desktop application after extracting
the bundled PXRD example. Automatic mode keeps the plot full-width; Manual mode
puts the controls in a resizable sidebar while retaining the preview height.

The screenshots and CSV export use the source at
[`a39ed1e`](https://github.com/unwatcher/Plot_Digitizer/commit/a39ed1ebe3a25a5fc67a2d31391344327eb24b2f),
which includes the top-left mode toggle and resizable manual sidebar. The
published v0.3.0-preview.1 Windows ZIPs predate this layout change. Follow
[Run from source](../README.md#run-from-source) to try the current layout.

## Automatic mode

![Full-width automatic workspace with reviewed axis markers and extracted curve overlay](images/automatic-mode.png)

## Manual mode

![Manual workspace with a controls sidebar and full-height plot preview](images/manual-mode.png)

## Try the workflow

1. Open `examples/patent_plots/us8198452b2-fig1.png`. Check the detected green
   plot-area boundary.
2. Choose **Auto-calibration**. Review each accepted label and its tick marker
   on both axes, then choose **Apply reviewed calibration**. This example
   accepts seven X ticks and two Y ticks, with linear scales. OCR proposals can
   differ by installed engine and language support.
3. Choose **Auto Extraction**. Inspect the highlighted candidate, then choose
   **Add selected curve**. Here Autopilot selects Normal scan and threshold 160.
4. Check the overlay. The top-left **Manual mode** button opens full-height
   controls on the left; the same button becomes **Automatic mode** and restores
   the full-width preview. Current calibration and extracted data are retained.
5. Open **Digitized Data** to review the calibrated X/Y values.
6. Choose **Export → Export CSV** and save the result. The app confirms that
   563 points were written.

## Actual output and provenance

The [recorded session's CSV](../examples/workflow-demo.csv) contains the actual
`dataset,x,y` output from the desktop exporter. X is 2Theta in degrees and Y
is intensity in counts for this PXRD example. Full floating-point values are
preserved. The desktop data table rounds values for display.

The [example metadata](../examples/workflow-demo.json) records the source
commit, input/output hashes, calibration point counts, extraction settings and
screenshot hashes and dimensions. This session uses a seven-tick X fit; the
separate [static PXRD reference](../examples/pxrd-extraction.json) uses a
two-tick X fit, so their exported coordinates differ slightly.

The source figure is US8198452B2, Fig. 1. See the
[example provenance and reuse note](../examples/README.md). The demonstration
is one reviewed example, not an independent accuracy or speed benchmark.

## Updating the screenshots

The images are unmodified native application captures from the same CSV-export
session. To update them, run the workflow above on the intended source revision,
capture the overlay in both modes, and verify the exported CSV. Keep local
paths out of published images and update the screenshots, CSV and metadata
together.
