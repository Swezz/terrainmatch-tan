# TerrainMatch-TAN methodology

## Scope

TerrainMatch-TAN is a research prototype for terrain patch retrieval. It
explores a positioning workflow for a GPS denied scenario, but it does not
ingest live aircraft, vehicle, or altimeter data and does not produce a
validated navigation fix.

The current local manifest contains 887 patches from 52 mapped regions. Each
parent patch is a georeferenced 200 by 200 pixel window derived from an SRTM
GeoTIFF. The active manifest records the source raster, pixel offsets, affine
transform, and patch metadata.

## Siamese CNN

The model uses a shared convolutional encoder to map two elevation samples into
128 dimensional, L2 normalized embeddings. The encoder has three
convolution, batch normalization, ReLU, and max pooling blocks, followed by
adaptive average pooling and two fully connected layers.

Training uses contrastive loss. Positive pairs are independently cropped from
the same parent patch; negative pairs use crops from different parent
patches. The default training configuration is 30 epochs, batch size 32,
Adam at a learning rate of 0.001, and 2 percent unit range Gaussian noise.
Training and validation are split by parent patch. The saved checkpoint
records 710 training patches, 177 validation patches, 11,360 training pairs,
and 1,416 validation pairs.

The CNN ranks map windows by cosine similarity. Similarity is a ranking score,
not a probability or calibrated confidence value.

## NCC baseline

The normalized cross correlation implementation compares overlapping
reference and observation arrays over candidate row and column offsets. It
returns the offset with the highest correlation score. The app runs NCC as a
separate synthetic comparison on the selected terrain sample; it does not
combine NCC with the CNN into a fused position estimate.

## Interactive demonstration

The Streamlit app requires the user to choose a test region and terrain sample.
It then simulates an observation by cropping and shifting the selected DEM
patch and adding configurable noise. The CNN search is limited to patches in
the selected region and scans candidate windows at a 32 pixel stride.

The highest ranked window is converted to a candidate coordinate using the
manifest's affine transform. The map views display that candidate and a
selected terrain waypoint. The terrain map is generated from local DEM data
when available, with the matched patch as a fallback. The street map uses
online CARTO tiles based on OpenStreetMap data. The line between candidate and
destination is a direct line, not a road or trail route.

## Saved synthetic evaluation

The saved CNN evaluation contains 708 simulated displaced queries from the
177 patch validation split, using four noise levels. Queries are compared with
sliding windows from the 887 patch reference gallery. The current saved result
is:

| Metric | Result |
|---|---:|
| CNN top 1 retrieval | 1.8% |
| CNN top 3 retrieval | 5.2% |
| Mean query time in the saved CPU run | 17.1 ms |

The saved NCC test reports 100 percent exact offset recovery on controlled
synthetic translations of the same terrain sample. This is a different task
from CNN retrieval and the results are not directly comparable.

These tests are pipeline checks, not validation on real sensor measurements or
geographically unseen terrain. The random patch split does not guarantee
geographic separation: nearby terrain windows can be spatially correlated.
The CNN results also show that the current model does not reliably localize
displaced observations, particularly with noise.

## Data acquisition

The repository includes example region lists and a downloader for
OpenTopography's Global Datasets API. A personal key is read from the
`OPENTOPOGRAPHY_API_KEY` environment variable and must not be committed or
shared. The patch builder discovers GeoTIFF files in `data/` and rewrites the
active patch manifest from those available rasters.

The active DEMs use NASA's SRTM Global dataset distributed by OpenTopography:
NASA Shuttle Radar Topography Mission (SRTM) (2013), *Shuttle Radar
Topography Mission (SRTM) Global*, https://doi.org/10.5069/G9445JDF (accessed
2026-10-25). OpenTopography asks projects using its API services to include
this acknowledgement: “This work is based on API services provided by the
OpenTopography Facility with support from the National Science Foundation
under NSF Award Numbers 2410799, 2410800 & 2410801.” Check the dataset landing
page for any dataset-specific license and citation details before
redistribution.

SRTM coverage is limited and does not include Antarctica. Adding more regions
to a list or downloading more rasters does not establish continuous coverage
or model accuracy.

## Limitations

- No live sensor input, vehicle telemetry, or field validation.
- Geographic search is limited to the user selected region.
- Current CNN retrieval performance is weak.
- Similarity scores are not probabilities and no calibrated uncertainty is
  produced.
- Synthetic Gaussian noise is not a validated altimeter error model.
- The street map requires internet access; it does not provide route planning.
- No safety, aviation, or operational use has been validated.
