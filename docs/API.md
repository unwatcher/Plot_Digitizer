# Local workflow API

Run the same Autopilot engine from a notebook, local application, ELN/LIMS adapter
or batch script. The API runs independently of the desktop window and uses the
existing NumPy/OpenCV dependencies and Python's standard library. No account,
API key or external service is needed.

## Start

In a Windows release folder, double-click **launch_api.bat**. It uses the bundled
Python and dependencies. The offline-wheels edition installs its verified local
wheels first when needed. Keep the console open; Ctrl+C stops the service.

From source, install `requirements-core.txt` and run:

~~~text
python -m plot_digitizer.api
~~~

The default socket is `http://127.0.0.1:8765`. Choose another port with
`python -m plot_digitizer.api --port 9000` or `launch_api.bat --port 9000`.
`--host ::1` selects IPv6 loopback. Public/LAN binding is outside this local
service's scope. A server-side adapter can call it on the same host.

## Endpoints

| Method | Endpoint | Result |
| --- | --- | --- |
| GET | `/health` | Readiness, API version and limits |
| GET | `/openapi.json` | OpenAPI 3.1 request contract |
| POST | `/v1/extract` | Curve candidates and diagnostics; JSON, CSV or XRDML |
| POST | `/v1/calibration` | Local-OCR tick/scale proposals for review |

Send `Content-Type: application/json`. Each request uploads PNG/JPEG bytes as
`image_base64` and receives the result directly. The API never loads a caller's
path or URL, and does not retain images or results. Save the response and source
image in your workflow when provenance matters.

## Try one image

With the server running, open another terminal in the source or release folder:

~~~text
python scripts/api_example.py examples/patent_plots/us8198452b2-fig1.png --output output/curves.json
~~~

The client uses only Python's standard library and any installed Python 3.
In a Windows bundle, replace `python` with `python314\python.exe`.
Use `--url http://127.0.0.1:9000` for another port, or add
`--suggest-calibration` to request OCR proposals alongside extraction.

Call the API from another Python program:

~~~python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

payload = {
    "image_base64": base64.b64encode(Path("plot.png").read_bytes()).decode("ascii"),
    "suggest_calibration": True,
}
request = Request(
    "http://127.0.0.1:8765/v1/extract",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
for curve in result["candidates"]:
    print(curve["id"], curve["label"], len(curve["points"]), curve["separation"])
~~~

## PowerShell quickstart

With the service running, open a second PowerShell terminal in the repository
root. This example uses only built-in PowerShell and .NET commands:

~~~powershell
$baseUrl = 'http://127.0.0.1:8765'
Invoke-RestMethod -Uri "$baseUrl/health"

$imagePath = (Resolve-Path 'examples/patent_plots/us8198452b2-fig1.png').Path
$payload = @{
    image_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
} | ConvertTo-Json

$result = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/extract" `
    -ContentType 'application/json' -Body $payload -TimeoutSec 120

New-Item -ItemType Directory -Path output -Force | Out-Null
$result | ConvertTo-Json -Depth 100 |
    Set-Content -Encoding utf8 output/curves-powershell.json

"Candidates: $(@($result.candidates).Count)"
"Coordinates: $($result.coordinate_system)"
~~~

`ConvertTo-Json` defaults to a depth of 2. Use `-Depth 100` when saving the
response so nested candidate objects, point arrays and diagnostics stay JSON
objects and arrays rather than becoming strings. The request here is shallow
and does not need that increased depth.

Without reviewed calibration, `coordinate_system` is `image_pixels` and
candidate `points` are source-image pixel coordinates, not scientific axis
values. Review the extracted curves and calibration before using numeric
results; see the calibration workflow below.

## Review calibration, then request numeric values

`points` always contains source-image pixels: the origin is at the top left and
Y increases downward. Encoded PNG/JPEG dimensions are used; EXIF orientation is
not applied. Without reviewed calibration, `coordinate_system` is `image_pixels`
and each candidate's `data_points` is null.

With `suggest_calibration: true`, extraction also returns `calibration_proposal`.
Alternatively, call `/v1/calibration` with `image_base64` and an optional `roi`.
Proposals can be partial (`ready: false`) when labels or a local OCR engine are
unavailable. They are never applied automatically.

Have your application display the source image/tick proposals for review, or
reuse calibration reviewed for the same image geometry. Here is an illustrative
`calibration.json`; replace the ticks and values with ones from your own image:

~~~json
{
  "reviewed": true,
  "axes": {
    "x": {"scale": "linear", "points": [
      {"pixel": [40, 190], "value": 5},
      {"pixel": [300, 190], "value": 45}
    ]},
    "y": {"scale": "linear", "points": [
      {"pixel": [40, 190], "value": 0},
      {"pixel": [40, 30], "value": 100}
    ]}
  }
}
~~~

Set `reviewed: true` after review and send this object as `calibration`.
Axes accept `linear` or `log`, including decreasing values, with 2–64 ticks each.
`data_points` then contains calibrated values and `coordinate_system` becomes
`axis_values`. Original pixel points remain available for overlays and auditing.

~~~text
python scripts/api_example.py plot.png --calibration calibration.json --format csv --output output/trace.csv
~~~

CSV uses `candidate_id,pixel_x,pixel_y,x,y`. The last two columns are blank without
calibration. JSON retains the extraction metadata.

## Request options and results

| Field | Meaning |
| --- | --- |
| `image_base64` | Required PNG/JPEG file bytes as base64; no data-URL prefix |
| `roi` | Optional [left, top, right, bottom] inside the source image; otherwise Autopilot detects the plot area |
| `suggest_calibration` | Request editable OCR proposals; default false |
| `calibration` | Reviewed X/Y calibration as above |
| `candidate_ids` | Optional array of zero-based IDs to keep; omitted means all |
| `format` | `json` (default), `csv`, or `xrdml` |
| `xrdml` | PXRD confirmation and export settings, described below |

JSON includes `source_image` (dimensions/SHA-256 of uploaded bytes), ROI, preset,
threshold/grid handling, threshold trials, overlap regions, coverage, separation
diagnostics and calibration inputs. IDs refer to the unfiltered candidate list
for this image and ROI; keep those inputs unchanged when selecting IDs in a
repeat request. No detected curves means an empty candidate list.

Only observed image-supported samples are returned. Missing columns are not
filled and curves are not smoothed. Coverage/separation diagnostics are
heuristics, not calibrated probabilities. Review curve identity where same-color
strokes cross, touch or overlap.

## XRDML for diffraction software

Use `format: "xrdml"`, reviewed calibration and one selected curve. Set
`xrdml.confirmed_pxrd: true` after confirming X is 2Theta in degrees and Y is
counts or counts per second. The client supports:

~~~text
python scripts/api_example.py pxrd.png --calibration calibration.json --candidate-id 0 --format xrdml --confirm-pxrd --output output/trace.xrdml
~~~

The response is XRDML 2.2 XML from the same exporter as the desktop app. The
`xrdml` object accepts `dataset_label`, `wavelength_ka1`, `wavelength_ka2`,
`wavelength_kb`, `generator_voltage`, `generator_current`, `counting_time`,
`y_scale` and `y_baseline`. Defaults are Cu wavelengths, 45 kV/40 mA, 1 second,
no scaling and no baseline subtraction. The API labels the result as digitized,
reconstructed data. Export settings do not recover original instrument metadata.

XRDML stores rounded nonnegative integer counts. For cps input use the default
`counting_time: 1`; for raw counts set the actual dwell time when known. The
current writer uses start/end angles, so the API rejects nonuniform sampling or
missing internal columns rather than changing the angles. JSON/CSV preserve
those samples. SPA export is not implemented.

## Local browser applications

Python/Node backends can call the socket directly. For a local web frontend,
enable its exact origin when starting the service:

~~~text
python -m plot_digitizer.api --allow-origin http://localhost:5173
~~~

Repeat `--allow-origin` for more local origins. CORS preflight supports GET/POST
and Content-Type. Other web origins are refused. Host headers are checked
against the bound loopback address/port.

## Limits and errors

Limits are 16 MiB of JSON, 12 MiB of image bytes, 16 megapixels and 16000 pixels
per side; the earliest limit applies. Dimensions are checked before decoding.
Two processing slots bound simultaneous image work. Local OCR uses installed
Windows language support or Tesseract; missing OCR can yield partial calibration
proposals without preventing pixel extraction.

Errors are JSON: `{"error":{"code":"invalid_request","message":"..."}}`.
Statuses include 400 (JSON), 403 (Host/Origin), 411 (Content-Length), 413 (size),
415 (media), 422 (plot/options/calibration), and 503 (busy; Retry-After: 1).
This is a synchronous loopback API for a trusted local host. It has no persistent
job store, shared projects or remote authentication layer.
