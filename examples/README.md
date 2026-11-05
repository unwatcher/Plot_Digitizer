# Patent plot provenance and reuse

This distribution contains four technical plot crops from two U.S. utility
patents. The selection was reviewed on 2026-11-04 against the
[USPTO patent-document reuse policy](https://www.uspto.gov/terms-use-uspto-websites).
That policy describes patent text and drawings as generally unrestricted, with
specific copyright/mask-work exceptions under 37 CFR 1.71(d/e) and 1.84(s).
See [MPEP 608.01(w)](https://www.uspto.gov/web/offices/pac/mpep/s608.html).

| Publication | Bundled figures | Source PDF checks |
| --- | --- | --- |
| [US8198452B2](https://patents.google.com/patent/US8198452B2/en) | 1 (PXRD), 4 (Raman), 7 (IR) | 25 pages; no copyright/mask-work/no-objection/all-rights-reserved notice found in extracted text |
| [US6583283B1](https://patents.google.com/patent/US6583283B1/en) | 8 (stacked Raman) | 53 pages; same notice search found no match |

The selected original drawing sheets and final crops were also visually checked.
No copyright notice or embedded brand/logo was found in the plot crops; the
instrument information tables below the spectra are outside the crops. Source
hashes, page numbers, rendering resolution, rotation and crop rectangles are in
`patent_plots/manifest.json`. Plots retain the scientific labels and axes.

This is a documented policy-based selection of U.S. patent drawings. The figures
are not relabeled CC0 or relicensed under the software BSD license. The review
does not assert a blanket worldwide public-domain status or grant rights to
practice any patented invention. Additions to this directory need their own
source and reuse review; public online availability alone is insufficient.

The four examples are correlated illustrative inputs from two families, not a
blind validation dataset. Numeric calibration, identity through touching curves
and piecewise axis scales require scientific review even when extraction checks
pass.
