# `.dvc/plots/templates/` — Vega-Lite Plot Templates for SpectraMind V50

Mission-grade, reusable **Vega-Lite v5** specs used by DVC (`dvc plots show|diff`) to render pipeline diagnostics
consistently across local runs, CI, and Kaggle. These templates are *schema-agnostic* except for the
expected column names (documented below).

> TL;DR  
> - Put your CSV/JSON diagnostics in `outputs/diagnostics/**.csv|.json`.  
> - Call `dvc plots show --targets .dvc/plots.yaml` (or a specific plot id).  
> - Choose light/dark variants to match your dashboard theme.

---

## Available templates

| File                      | Best for                                                                 | X (field) | Y (field) | Color (field) | Notes |
|--------------------------|--------------------------------------------------------------------------|-----------|-----------|----------------|-------|
| `line.json`              | Time/step curves (loss, LR, GLL, metric vs step/epoch)                   | `step`    | `value`   | `rev`          | Points + 2px line; grid on. |
| `line_dark.json`         | Same as above on dark dashboards                                         | `step`    | `value`   | `rev`          | Dark background; light labels. |
| `scatter_guide.json`     | Calibration: \|residual\| vs σ (or any x–y scatter with y=x guide)       | `x`       | `y`       | `rev`          | Dashed identity line; filled points. |
| `hist.json`              | Distributions (residuals, σ, errors, scores)                             | `value`   | count     | `rev`          | 40 maxbins; semi-transparent bars. |
| `heatmap.json`           | Per-bin/per-metric matrices (GLL heatmap, coverage, violations)          | `bin`     | `metric`  | `value`        | Viridis colormap, light theme. |
| `heatmap_dark.json`      | Same heatmap on dark dashboards                                          | `bin`     | `metric`  | `value`        | Plasma colormap; dark background. |

> **Tip:** `rev` is the label DVC injects for each revision/run (branch/commit/workspace), enabling instant
comparisons across runs in the same figure.

---

## Expected data columns (minimal)

To keep templates reusable, we assume tidy/long format with these canonical columns:

- **Curves (line/\*_dark)**  
  - `step` *(quantitative)*: training step / epoch / wall index  
  - `value` *(quantitative)*: metric value (loss, LR, GLL, etc.)  
  - `rev` *(nominal)*: DVC revision (auto from DVC)

- **Scatter (scatter_guide.json)**  
  - `x` *(quantitative)*: predicted σ (or independent var)  
  - `y` *(quantitative)*: \|μ−y\| (or dependent var)  
  - `rev` *(nominal)*: DVC revision

- **Histogram (hist.json)**  
  - `value` *(quantitative)*: sample values to bin (residuals/σ/etc.)  
  - `rev` *(nominal)*: DVC revision

- **Heatmaps (heatmap/\*_dark)**  
  - `bin` *(ordinal|nominal)*: wavelength bin (1..283)  
  - `metric` *(ordinal|nominal)*: metric name/key (e.g., `GLL`, `coverage@90`, `violations`)  
  - `value` *(quantitative)*: cell intensity/value

> If your column names differ, map/rename upstream or adapt `.dvc/plots.yaml` to transform fields.

---

## How to use with DVC

### 1) Put your diagnostics where DVC (and you) expect them
```

outputs/
└─ diagnostics/
├─ train\_metrics.json          # step,value,rev
├─ val\_metrics.json            # step,value,rev
├─ residuals.csv               # value,rev
├─ calibration.csv             # x,y,rev
└─ gll\_per\_bin.csv             # bin,metric,value,rev

````

### 2) Reference these templates from `.dvc/plots.yaml`
Make sure your YAML points at these paths and chooses a template id (examples shown in our project’s file):

```yaml
version: 1
templates:
  line:            plots/templates/line.json
  line_dark:       plots/templates/line_dark.json
  scatter_guide:   plots/templates/scatter_guide.json
  hist:            plots/templates/hist.json
  heatmap:         plots/templates/heatmap.json
  heatmap_dark:    plots/templates/heatmap_dark.json
plots:
  - id: train_loss
    title: "Training Loss vs Step"
    path: outputs/diagnostics/train_metrics.json
    template: line
  # ... add more plots here ...
````

### 3) Render locally

```bash
# All plots defined in .dvc/plots.yaml
dvc plots show --targets .dvc/plots.yaml

# Or a single plot id:
dvc plots show --targets .dvc/plots.yaml:train_loss

# Compare runs (current vs HEAD~1)
dvc plots diff --targets .dvc/plots.yaml HEAD~1
```

DVC will open an interactive HTML with embedded Vega-Lite. You can also save with `--out` if needed.

---

## Theming & accessibility

* **Light vs Dark:** Choose `*_dark.json` variants for dark dashboards (CLI reports, night mode).
* **Colormaps:**

  * **Viridis/Plasma** are perceptually uniform and color-blind friendly—suitable for science-grade figures.
* **Labels:** Heatmaps use rotated x-labels (bins) for dense axes; adjust in the template if your bins are sparse.

---

## Performance & data hygiene

* Prefer **long/tidy** data: easier to facet/filter and combine across `rev`.
* Downsample very dense curves (e.g., log every N steps) to keep pages responsive.
* For heatmaps, keep `bin` and `metric` categorical/ordinal; avoid free-text typos (use controlled vocab).

---

## Adapting templates

These are plain Vega-Lite specs. You can:

* Change color schemes: set `"scale": {"scheme": "magma"}` or custom `"range"` arrays.
* Add tooltips: extend `mark.tooltip` or field tooltips in `encoding`.
* Switch axes to quantitative if your bins are numeric and continuous.

> If you modify templates, keep variants in this folder and update the `templates:` section of `.dvc/plots.yaml`.

---

## FAQ

**Q: My plot renders but axes are wrong.**
A: Check field **types** (`quantitative|ordinal|nominal`) and column names. Vega-Lite will guess, but explicit is better.

**Q: How do I compare two runs on the same chart?**
A: DVC adds a `rev` column—our templates color by `rev` so multiple runs appear side-by-side automatically.

**Q: Can I export static PNGs?**
A: Use the DVC HTML download button or open the HTML in a modern browser and print/save; for batch export, use `vl-convert` or headless Chromium.

---

## Changelog

* **v1.0** — Initial set: `line`, `line_dark`, `scatter_guide`, `hist`, `heatmap`, `heatmap_dark`.

---

```
```
