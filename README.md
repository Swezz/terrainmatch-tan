# TerrainMatch-TAN

**A terrain matching research prototype for GPS denied positioning**

TerrainMatch-TAN explores whether terrain elevation patterns can help estimate
a candidate location when GPS is unavailable. It combines a Siamese
convolutional neural network (CNN), a normalized cross correlation (NCC)
baseline, a local terrain view, and a standard street map.

> **Research prototype only.** The app uses simulated terrain observations and
> a limited, preloaded DEM library. It is not a live positioning system, a
> verified navigation aid, or safe for operational use.

## Project highlights

- Builds georeferenced terrain patches from SRTM GeoTIFF elevation data.
- Trains and runs a Siamese CNN that ranks candidate terrain windows.
- Includes NCC as a separate, classical comparison method.
- Demonstrates the workflow in a Streamlit interface with simulation controls,
  candidate coordinates, a terrain view, and a street map.
- Includes scripts for data ingestion, training, evaluation, and a command line
  demonstration.

## How the demo works

1. Select a test region from the local terrain library.
2. The app creates a simulated observation from that region's DEM patch.
3. The CNN ranks candidate windows within the selected region.
4. The highest ranked candidate is displayed with its mapped coordinates.
5. NCC runs as a separate cross check on a synthetic translation of the
   selected terrain sample.
6. The result is shown on a local terrain view and an online street map.

The app does **not** receive live altimeter or vehicle data. The region choice
limits the search; it does not automatically identify a user's region. The
line between the candidate and selected destination is a straight line, not a
road or trail route.

The terrain view uses a local source DEM when available and falls back to the
matched terrain patch. The street map is provided by CARTO, uses OpenStreetMap
data, and requires internet access. Attribution links appear below each map.

## Technology

| Area | Tools |
|---|---|
| Model | PyTorch, Siamese CNN, contrastive loss |
| Matching baseline | NumPy based normalized cross correlation |
| Geospatial data | SRTM GeoTIFF, Rasterio, affine coordinate transforms |
| Interface | Streamlit, Matplotlib, PyDeck |
| Tests | pytest |

For the detailed model, data, and evaluation methodology, see
[docs/methodology.md](docs/methodology.md).

## Data and current scope

The current local dataset manifest contains **887 terrain windows from 52
mapped regions**. The demo needs the active manifest, its 887 derived patches,
and `models/terrain_cnn.pt` for CNN inference without retraining. The source
rasters in `data/` support rebuilding the patch library and detailed terrain
views; the app can use a matched-patch fallback when a source raster is
unavailable. The proposed repository is about 200 MiB, larger than a
source-only project. The library contains selected areas in India and Nepal
and other countries, not continuous worldwide coverage.

The terrain data is derived from SRTM and is approximately 90 m resolution
where available. SRTM does not cover Antarctica. For new areas, the project
includes example region lists and a downloader for OpenTopography. API access,
coverage, availability, and usage limits depend on OpenTopography's current
terms and account limits.

The OpenTopography API key is read from the `OPENTOPOGRAPHY_API_KEY`
environment variable. Keep it private. Do not put it in this repository, a
README, a screenshot, or a public issue.

## Quick start

### Requirements

- Windows, macOS, or Linux
- Python 3.11 recommended
- Internet access for the street map and optional DEM downloads

### Install and run on Windows

From PowerShell in the cloned repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`. Keep the PowerShell window running while using the
app. Stop the app with `Ctrl+C`.

If `py -3.11` is not available, install Python 3.11 and reopen PowerShell.
Activation of the virtual environment is optional because the commands above
call its Python executable directly.

### Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Optional: download more terrain data

The included manifest and patches are enough to try the demo. To request
additional DEMs from OpenTopography, first set your personal key in the current
PowerShell session only:

```powershell
$env:OPENTOPOGRAPHY_API_KEY = Read-Host "OpenTopography API key"
```

Then download the example region list in batches:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_opentopography_dem.py `
  --regions-file data\global_regions.example.json `
  --start-index 1 --count 10
```

Repeat with `--start-index 11`, `21`, `31`, and `41` to request the remaining
areas. Successful downloads are retained if a request fails. Build a new patch
manifest from the GeoTIFF files in `data/`:

```powershell
.\.venv\Scripts\python.exe scripts\build_terrain_dataset.py `
  --patches-per-dem 40 `
  --minimum-std-m 0
```

The builder replaces the active manifest with entries generated from the
available rasters. Rebuilding a larger dataset can take time and increases
storage use. The dataset manifest determines which patches the app searches.
Do not share or commit your API key.

## Training and evaluation

Train a CNN checkpoint using the active manifest:

```powershell
.\.venv\Scripts\python.exe -m terrainmatch.train --epochs 30
```

Run the synthetic CNN and NCC evaluations:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cnn.py
```

Training, evaluation, and CLI options are available with `--help`. Evaluation
outputs are written under `outputs/results/` and `outputs/figures/`.

## Benchmark results

The saved evaluation in `outputs/results/evaluation_summary.json` uses **887
DEM-derived patches across 52 regions**. The
CNN benchmark uses 708 simulated displaced queries from a random 80/20
patch level split. The candidate gallery includes windows from the stored
reference patches. Results are a pipeline benchmark, not an independent test
on new geographic regions or real sensor data.

| Method | Test | Result |
|---|---|---:|
| CNN | Top 1 retrieval over all queries | 1.8% |
| CNN | Top 3 retrieval over all queries | 5.2% |
| CNN | Mean query time in the saved CPU run | 17.1 ms |
| NCC | Exact offset recovery on controlled synthetic translations | 100% |

The CNN scores show that the current model does not reliably recover displaced
positions, especially with added noise. The NCC result measures a different,
controlled task on synthetic translations of the same terrain sample; it is
not comparable to CNN retrieval and does not indicate real world accuracy.
Neither result is evidence of safe or operational navigation performance.

## Repository layout

```text
app.py                       Streamlit demonstration
terrainmatch/                Dataset, model, training, inference, matching
scripts/                     Data preparation and evaluation commands
tests/                       Unit and data ingestion tests
data/                        Source DEMs and example region lists
outputs/terrain_dataset/     Derived patches and active manifest
outputs/results/             Saved evaluation summaries and tables
outputs/figures/              Evaluation and demo figures
models/                       Trained prototype checkpoint
docs/                         Methodology and coordinate conventions
demo_cli.py                   Command line demonstration
```

## Limitations and next steps

- Position estimates are simulated patch matches, not fixes from a live
  navigation sensor.
- The CNN's current retrieval results are weak and need improvement.
- A selected region is required; the app does not perform a global search
  across every place on Earth.
- The dataset is geographically sparse and does not provide continuous map
  coverage.
- The evaluation split is random at the patch level. Nearby windows from the
  same source DEM may be spatially correlated.
- The simulation uses configurable noise and offsets, not a validated sensor
  error model.
- The street map is online and its line is not a route. The local terrain view
  works independently of street map tiles.
- The prototype has not been validated with field data, aircraft telemetry,
  real altimeter readings, or safety testing.

Promising next work includes geographically separated evaluation, realistic
sensor error models, better trained matching models, and integration with
verified route data. These are future directions, not current capabilities.

## Data attribution

The elevation samples use the SRTM Global dataset distributed by
OpenTopography:

> NASA Shuttle Radar Topography Mission (SRTM) (2013). *Shuttle Radar
> Topography Mission (SRTM) Global*. Distributed by OpenTopography.
> [https://doi.org/10.5069/G9445JDF](https://doi.org/10.5069/G9445JDF).
> Accessed 2026-10-25.

OpenTopography requests this acknowledgement when its API services are used in
research or open-source software: “This work is based on API services provided
by the OpenTopography Facility with support from the National Science
Foundation under NSF Award Numbers 2410799, 2410800 & 2410801.” Check the
dataset landing page for applicable data licensing and citation details before
redistributing data.

Street maps use CARTO basemaps and OpenStreetMap data. The app displays links
to [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and
[© CARTO](https://carto.com/attribution/) below each map.

## Disclaimer

This project is for research, education, and demonstration only. It is not
certified navigation software, does not provide emergency guidance, and must
not be used to make safety critical decisions.
