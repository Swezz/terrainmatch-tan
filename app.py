"""Streamlit interface for the TerrainMatch-TAN positioning demonstration."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print(
        "ERROR: Streamlit is not installed.\n"
        "Install it with:\n"
        "    .venv\\Scripts\\pip.exe install streamlit\n"
        "Then launch with:\n"
        "    .venv\\Scripts\\streamlit.exe run app.py"
    )
    sys.exit(1)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from PIL import Image

from terrainmatch.dataset import (
    CROP_SIZE,
    DATASET_DIR,
    MANIFEST_PATH,
    load_manifest,
    load_patch,
    normalize_patch,
)
from terrainmatch.inference import (
    DEFAULT_CHECKPOINT,
    build_gallery,
    load_model,
    match_patch,
)
from terrainmatch.matching import sliding_window_ncc
from terrainmatch.simulation import simulate_observation

EVALUATION_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "results" / "evaluation_summary.json"

st.set_page_config(page_title="TerrainMatch-TAN", layout="wide")


@st.cache_data
def _load_manifest(manifest_mtime_ns: int):
    if MANIFEST_PATH.stat().st_mtime_ns != manifest_mtime_ns:
        raise RuntimeError("The dataset manifest changed while the app was loading.")
    return load_manifest(MANIFEST_PATH)


@st.cache_resource
def _load_model_cached(checkpoint_path: str, checkpoint_mtime_ns: int):
    path = Path(checkpoint_path)
    if path.stat().st_mtime_ns != checkpoint_mtime_ns:
        raise RuntimeError("The CNN checkpoint changed while the app was loading.")
    return load_model(path)


@st.cache_resource
def _load_gallery_cached(
    checkpoint_path: str,
    checkpoint_mtime_ns: int,
    manifest_mtime_ns: int,
):
    model = _load_model_cached(checkpoint_path, checkpoint_mtime_ns)
    embeddings, items = build_gallery(
        _load_manifest(manifest_mtime_ns), model, DATASET_DIR, stride=32
    )
    return model, embeddings, items


@st.cache_data
def _load_patch_data(file_name: str):
    manifest = _load_manifest(MANIFEST_PATH.stat().st_mtime_ns)
    item = next((entry for entry in manifest if entry["file"] == file_name), None)
    if item is None:
        return None, None
    return load_patch(item, DATASET_DIR), item


def _display_region(region: str) -> str:
    """Format dataset identifiers as readable region names."""
    readable = region.replace("_", " ").replace("-", " ")
    if readable.lower().endswith(" srtmgl3"):
        readable = readable[:-8]
    if readable.lower().startswith("srtm "):
        readable = readable[5:]
    return readable.title()


def _terrain_figure(arr: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(arr, cmap="terrain", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Relative elevation")
    fig.tight_layout()
    return fig


def _coordinates_at(item: dict, row: float, col: float) -> tuple[float, float] | None:
    transform = item.get("transform")
    if transform is None or len(transform) < 6:
        return None
    a, b, c, d, e, f = (float(value) for value in transform[:6])
    longitude = a * col + b * row + c
    latitude = d * col + e * row + f
    return latitude, longitude


def _distance_and_bearing(
    start: tuple[float, float],
    destination: tuple[float, float],
) -> tuple[float, float]:
    """Return the great circle distance in kilometres and initial bearing."""
    start_lat, start_lon = map(math.radians, start)
    end_lat, end_lon = map(math.radians, destination)
    delta_lat = end_lat - start_lat
    delta_lon = end_lon - start_lon

    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(start_lat) * math.cos(end_lat) * math.sin(delta_lon / 2) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    distance_km = 6371.0088 * 2 * math.asin(math.sqrt(haversine))
    y = math.sin(delta_lon) * math.cos(end_lat)
    x = (
        math.cos(start_lat) * math.sin(end_lat)
        - math.sin(start_lat) * math.cos(end_lat) * math.cos(delta_lon)
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    return distance_km, bearing


def _bounds_from_transform(
    transform: list[float] | tuple[float, ...],
    width: int,
    height: int,
) -> list[float]:
    """Convert a six or nine value affine transform to west, south, east, north."""
    a, b, c, d, e, f = (float(value) for value in transform[:6])
    corners = [
        (c, f),
        (a * width + c, d * width + f),
        (b * height + c, e * height + f),
        (a * width + b * height + c, d * width + e * height + f),
    ]
    longitudes, latitudes = zip(*corners)
    return [
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
    ]


@st.cache_data(max_entries=8)
def _load_offline_terrain_map(
    source_dem: str,
    fallback_patch_file: str,
    fallback_transform: tuple[float, ...],
) -> tuple[Path, list[float], bool, str | None]:
    """Build an inline terrain image so the map does not depend on online tiles."""
    rasterio_import_error = None
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        rasterio = None
        rasterio_import_error = str(exc)

    dem_path = PROJECT_ROOT / "data" / source_dem
    if dem_path.is_file() and rasterio is not None:
        with rasterio.open(dem_path) as dem:
            if dem.crs is None or dem.crs.to_epsg() != 4326:
                raise ValueError(
                    f"The offline map requires a DEM in EPSG:4326: {dem_path}"
                )
            scale = min(1.0, 700 / max(dem.height, dem.width))
            height = max(1, round(dem.height * scale))
            width = max(1, round(dem.width * scale))
            elevation = dem.read(
                1,
                out_shape=(height, width),
                masked=True,
                resampling=Resampling.average,
            )
            elevation = np.asarray(
                elevation.astype(np.float32).filled(np.nan),
                dtype=np.float32,
            )
            bounds = [
                dem.bounds.left,
                dem.bounds.bottom,
                dem.bounds.right,
                dem.bounds.top,
            ]
            has_full_dem = True
    else:
        manifest = _load_manifest(MANIFEST_PATH.stat().st_mtime_ns)
        fallback_item = next(
            (item for item in manifest if item["file"] == fallback_patch_file),
            None,
        )
        if fallback_item is None:
            raise ValueError(
                f"Terrain map patch is missing from the active manifest: "
                f"{fallback_patch_file}"
            )
        elevation = load_patch(fallback_item, DATASET_DIR)
        bounds = _bounds_from_transform(
            fallback_transform,
            elevation.shape[1],
            elevation.shape[0],
        )
        has_full_dem = False

    valid = np.isfinite(elevation)
    if not valid.any():
        raise ValueError(f"No valid elevation cells were found for {source_dem}.")
    low, high = np.percentile(elevation[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((elevation - low) / (high - low), 0, 1)
    rgba = (plt.get_cmap("terrain")(scaled) * 255).astype(np.uint8)
    rgba[~valid, 3] = 0
    image = Image.fromarray(rgba)
    image_key = hashlib.sha256(
        f"{source_dem}:{fallback_patch_file}".encode("utf-8")
    ).hexdigest()[:16]
    image_path = Path(tempfile.gettempdir()) / f"terrainmatch_map_{image_key}.png"
    image.save(image_path, format="PNG", optimize=True)
    return image_path, bounds, has_full_dem, rasterio_import_error


def _offline_route_figure(
    image_path: Path,
    bounds: list[float],
    origin: tuple[float, float],
    destination: tuple[float, float],
    zoom: float,
) -> plt.Figure:
    """Draw an offline terrain map with route markers fixed to screen size."""
    origin_lat, origin_lon = origin
    destination_lat, destination_lon = destination
    raster = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111820")
    ax.set_facecolor("#111820")
    ax.imshow(
        raster,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        origin="upper",
        interpolation="bilinear",
        aspect="auto",
    )
    ax.plot(
        [origin_lon, destination_lon],
        [origin_lat, destination_lat],
        color="#2BC98A",
        linewidth=2.5,
        zorder=3,
    )
    ax.scatter(
        [origin_lon, destination_lon],
        [origin_lat, destination_lat],
        s=[360, 300],
        c=["#E88B24", "#19B879"],
        edgecolors=["white", "#E8FFF4"],
        linewidths=2,
        zorder=4,
    )
    ax.annotate(
        "CANDIDATE LOCATION",
        (origin_lon, origin_lat),
        xytext=(10, 12),
        textcoords="offset points",
        color="white",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#202833", "alpha": 0.95},
        zorder=5,
    )
    ax.annotate(
        "DESTINATION",
        (destination_lon, destination_lat),
        xytext=(10, 12),
        textcoords="offset points",
        color="white",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#202833", "alpha": 0.95},
        zorder=5,
    )
    middle_lon = (origin_lon + destination_lon) / 2
    middle_lat = (origin_lat + destination_lat) / 2
    longitude_span = bounds[2] - bounds[0]
    latitude_span = bounds[3] - bounds[1]
    view_width = longitude_span / zoom
    view_height = latitude_span / zoom
    ax.set_xlim(middle_lon - view_width / 2, middle_lon + view_width / 2)
    ax.set_ylim(middle_lat - view_height / 2, middle_lat + view_height / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", color="#D1DAE3")
    ax.set_ylabel("Latitude", color="#D1DAE3")
    ax.tick_params(colors="#D1DAE3")
    for spine in ax.spines.values():
        spine.set_color("#758393")
    fig.tight_layout()
    return fig


def _offline_preview_figure(
    image_path: Path,
    bounds: list[float],
    sample: tuple[float, float] | None,
) -> plt.Figure:
    """Show the selected DEM area before a simulated position is estimated."""
    raster = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111820")
    ax.set_facecolor("#111820")
    ax.imshow(
        raster,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        origin="upper",
        interpolation="bilinear",
        aspect="auto",
    )
    if sample is not None:
        sample_lat, sample_lon = sample
        ax.scatter(
            [sample_lon],
            [sample_lat],
            s=300,
            c=["#7186A5"],
            edgecolors="white",
            linewidths=2,
            zorder=4,
        )
        ax.annotate(
            "SELECTED SAMPLE",
            (sample_lon, sample_lat),
            xytext=(10, 12),
            textcoords="offset points",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#202833", "alpha": 0.95},
            zorder=5,
        )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", color="#D1DAE3")
    ax.set_ylabel("Latitude", color="#D1DAE3")
    ax.tick_params(colors="#D1DAE3")
    for spine in ax.spines.values():
        spine.set_color("#758393")
    fig.tight_layout()
    return fig


def _standard_map(
    origin: tuple[float, float],
    destination: tuple[float, float] | None = None,
) -> pdk.Deck:
    """Build a standard street map with candidate and optional destination."""
    origin_lat, origin_lon = origin
    markers = [
        {
            "position": [origin_lon, origin_lat],
            "color": [232, 139, 36],
            "label": "Candidate location",
        }
    ]
    layers = []
    center_lat, center_lon = origin_lat, origin_lon
    zoom = 11

    if destination is not None:
        destination_lat, destination_lon = destination
        markers.append(
            {
                "position": [destination_lon, destination_lat],
                "color": [25, 184, 121],
                "label": "Destination",
            }
        )
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[
                    {
                        "path": [
                            [origin_lon, origin_lat],
                            [destination_lon, destination_lat],
                        ],
                        "color": [34, 135, 101],
                    }
                ],
                get_path="path",
                get_color="color",
                get_width=100,
                width_min_pixels=3,
            )
        )
        center_lat = (origin_lat + destination_lat) / 2
        center_lon = (origin_lon + destination_lon) / 2
        distance_km, _ = _distance_and_bearing(origin, destination)
        zoom = max(6, min(14, 14 - math.log2(max(distance_km, 1))))

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position="position",
            get_fill_color="color",
            get_radius=400,
            radius_min_pixels=9,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{label}"},
        height=480,
        description="Street map showing the candidate location and destination",
    )


st.sidebar.title("TerrainMatch-TAN")
st.sidebar.caption("Terrain based positioning when GPS is unavailable")
st.sidebar.caption("Switch themes in the app menu under Settings > Appearance.")

manifest_mtime_ns = MANIFEST_PATH.stat().st_mtime_ns
manifest = _load_manifest(manifest_mtime_ns)
if not manifest:
    st.error("The terrain library is empty. Build the dataset before starting the demo.")
    st.stop()

regions = sorted({item["region"] for item in manifest}, key=_display_region)
region_labels = {_display_region(region): region for region in regions}
preferred_region = "Sikkim Gangtok"
preferred_index = (
    list(region_labels).index(preferred_region)
    if preferred_region in region_labels
    else 0
)

st.title("Find your position when GPS is unavailable.")
st.markdown(
    "Choose the area you think you are in. The CNN will compare a simulated "
    "terrain reading with map data available for that area."
)

selected_region_label = st.selectbox(
    "Where do you think you are?",
    list(region_labels),
    index=preferred_index,
    help="The search is limited to mapped terrain in the selected area.",
)
selected_region = region_labels[selected_region_label]
region_items = [item for item in manifest if item["region"] == selected_region]
patch_labels = {
    item["file"]: f"Sample {index:02d}"
    for index, item in enumerate(region_items, start=1)
}
with st.sidebar.expander("Simulation settings"):
    selected_patch_file = st.selectbox(
        "Terrain sample",
        list(patch_labels),
        format_func=patch_labels.get,
    )
    noise_std = st.slider("Observation noise", 0.0, 0.3, 0.05, 0.01)
    row_disp = st.slider("North to south offset (pixels)", 0, 100, 20)
    col_disp = st.slider("East to west offset (pixels)", 0, 100, 20)
    top_k = st.slider("CNN candidates to display", 3, 10, 5)

reference_patch, ref_item = _load_patch_data(selected_patch_file)
if reference_patch is None or ref_item is None:
    st.error("The selected terrain tile could not be loaded.")
    st.stop()

scenario_key = (
    selected_patch_file,
    noise_std,
    row_disp,
    col_disp,
    top_k,
    manifest_mtime_ns,
)

run_demo = st.button(
    "Run simulated position check",
    type="primary",
    width="stretch",
)

if run_demo:
    checkpoint_path = DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        st.error(
            f"CNN checkpoint not found at `{checkpoint_path}`. "
            "Train the model before running the positioning demo."
        )
    else:
        rng = np.random.default_rng(time.time_ns() % 2**32)
        observation, actual_row, actual_col = simulate_observation(
            reference_patch,
            noise_std=noise_std * (reference_patch.max() - reference_patch.min()),
            row_displacement=row_disp,
            col_displacement=col_disp,
            crop_size=CROP_SIZE,
            rng=rng,
        )
        with st.spinner("Searching the terrain library with the CNN..."):
            started_at = time.perf_counter()
            model, gallery_embeddings, gallery_items = _load_gallery_cached(
                str(checkpoint_path),
                checkpoint_path.stat().st_mtime_ns,
                manifest_mtime_ns,
            )
            region_gallery_indices = [
                index
                for index, item in enumerate(gallery_items)
                if item["region"] == selected_region
            ]
            if not region_gallery_indices:
                st.error(
                    f"No searchable terrain windows are available for "
                    f"{_display_region(selected_region)}."
                )
                st.stop()
            region_gallery_embeddings = gallery_embeddings[region_gallery_indices]
            region_gallery_items = [
                gallery_items[index] for index in region_gallery_indices
            ]
            ranked = match_patch(
                observation,
                region_gallery_embeddings,
                region_gallery_items,
                model,
                top_k=top_k,
            )
            cnn_elapsed_ms = (time.perf_counter() - started_at) * 1000

        ncc_started_at = time.perf_counter()
        ncc_result = sliding_window_ncc(reference_patch, observation)
        ncc_elapsed_ms = (time.perf_counter() - ncc_started_at) * 1000
        st.session_state["positioning_result"] = {
            "key": scenario_key,
            "observation": observation,
            "actual_row": actual_row,
            "actual_col": actual_col,
            "ranked": ranked,
            "ncc_result": ncc_result,
            "cnn_elapsed_ms": cnn_elapsed_ms,
            "ncc_elapsed_ms": ncc_elapsed_ms,
        }

map_reader_error = None
result = st.session_state.get("positioning_result")
if result is not None and result["key"] == scenario_key:
    observation = result["observation"]
    ranked = result["ranked"]
    best = ranked[0]
    best_patch, _ = _load_patch_data(best.patch_file)

    st.markdown("---")
    st.caption("Status: Candidate match found")
    st.header(f"Best matching location in {_display_region(selected_region)}")
    st.info(
        "This is a candidate from a simulated terrain reading, not a verified "
        "live position."
    )
    estimate_metrics = st.columns(2)
    estimate_metrics[0].metric(
        "Estimated latitude",
        f"{best.latitude:.3f}°" if best.latitude is not None else "Unavailable",
    )
    estimate_metrics[1].metric(
        "Estimated longitude",
        f"{best.longitude:.3f}°" if best.longitude is not None else "Unavailable",
    )

    best_item = _load_patch_data(best.patch_file)[1]
    if best.latitude is None or best.longitude is None or best_item is None:
        st.warning("Map coordinates are not available for this candidate.")
    else:
        destination_items = []
        for item in region_items:
            if item["source_dem"] != best_item["source_dem"]:
                continue
            waypoint = _coordinates_at(
                item,
                item["patch_size"] / 2,
                item["patch_size"] / 2,
            )
            if waypoint is None:
                continue
            distance_km, _ = _distance_and_bearing(
                (best.latitude, best.longitude), waypoint
            )
            if distance_km >= 0.1:
                destination_items.append((item, waypoint))

        if destination_items:
            destination_labels = {
                item["file"]: (
                    f"Terrain waypoint {index:02d}  ·  "
                    f"{waypoint[0]:.4f}°, {waypoint[1]:.4f}°"
                )
                for index, (item, waypoint) in enumerate(destination_items, start=1)
            }
            destination_files = list(destination_labels)
            selected_destination_file = st.selectbox(
                "Where would you like to go?",
                destination_files,
                format_func=destination_labels.get,
                key=f"destination_{selected_region}",
                help=(
                    "Choose a mapped point stored with this terrain area. "
                    "Place names and roads are not available offline."
                ),
            )
            destination_item = next(
                item
                for item, _ in destination_items
                if item["file"] == selected_destination_file
            )
            destination = next(
                waypoint
                for item, waypoint in destination_items
                if item["file"] == selected_destination_file
            )
            distance_km, bearing = _distance_and_bearing(
                (best.latitude, best.longitude), destination
            )
            compass_points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
            compass_direction = compass_points[int((bearing + 22.5) // 45) % 8]
            distance_label = (
                f"{distance_km * 1000:.0f} m"
                if distance_km < 1
                else f"{distance_km:.1f} km"
            )

            (
                image_uri,
                map_bounds,
                has_full_dem,
                map_reader_error,
            ) = _load_offline_terrain_map(
                best_item["source_dem"],
                best_item["file"],
                tuple(best_item["transform"]),
            )
            longitude_span = map_bounds[2] - map_bounds[0]
            latitude_span = map_bounds[3] - map_bounds[1]
            route_longitude_span = max(
                abs(best.longitude - destination[1]),
                longitude_span / 100,
            )
            route_latitude_span = max(
                abs(best.latitude - destination[0]),
                latitude_span / 100,
            )
            max_zoom = max(
                1.0,
                min(
                    5.0,
                    longitude_span / (route_longitude_span * 1.2),
                    latitude_span / (route_latitude_span * 1.2),
                ),
            )
            max_zoom = math.floor(max_zoom * 10) / 10
            st.subheader("Map and direction")
            if max_zoom > 1:
                map_zoom = st.slider(
                    "Map zoom",
                    min_value=1.0,
                    max_value=max_zoom,
                    value=1.0,
                    step=0.1,
                    key=f"map_zoom_{selected_region}",
                )
            else:
                map_zoom = 1.0
            route_figure = _offline_route_figure(
                image_uri,
                map_bounds,
                (best.latitude, best.longitude),
                destination,
                map_zoom,
            )
            st.markdown("**Terrain view**")
            st.pyplot(
                route_figure,
                clear_figure=True,
                width="stretch",
                alt=(
                    "Terrain map with the candidate location, destination, "
                    "and direct line between them"
                ),
            )
            plt.close(route_figure)
            st.markdown("**Street map**")
            st.pydeck_chart(
                _standard_map(
                    (best.latitude, best.longitude),
                    destination,
                ),
                width="stretch",
                height=480,
            )
            st.markdown(
                "[© OpenStreetMap contributors](https://www.openstreetmap.org/copyright)"
                " · [© CARTO](https://carto.com/attribution/)"
            )
            if not has_full_dem:
                st.warning(
                    "The full elevation map is unavailable here. The terrain "
                    "view shows the real matched terrain sample instead."
                )
            st.caption(
                "Orange marks the candidate; green marks the destination. "
                "This street map needs an internet connection."
            )
            guidance_cols = st.columns(2)
            guidance_cols[0].metric("Straight line distance", distance_label)
            guidance_cols[1].metric(
                "Direction from candidate",
                f"{bearing:.0f}° {compass_direction}",
            )
            st.warning(
                "This is a direct line between two mapped points. It does not "
                "follow roads or trails, avoid obstacles, or provide emergency "
                "guidance. Do not use it as a safe route."
            )
        else:
            st.info(
                "There are no separate mapped waypoints in this terrain area "
                "for a direction guide."
            )

    observation_col, match_col = st.columns(2)
    with observation_col:
        st.subheader("Simulated onboard sample")
        st.pyplot(
            _terrain_figure(normalize_patch(observation), "Elevation observation"),
            width="stretch",
        )
    with match_col:
        st.subheader("Best matching map window")
        if best_patch is not None:
            st.pyplot(
                _terrain_figure(
                    normalize_patch(best_patch),
                    _display_region(best.region),
                ),
                width="stretch",
            )
        st.caption(
            f"Map window: row {best.window_row}, column {best.window_col}. "
            "Similarity is a ranking score, not a probability."
        )
elif result is None:
    st.caption("Status: Ready")
    st.info(
        "Choose the area you believe you are near, then run a simulated "
        "position check. The search uses map data from that area only."
    )
else:
    st.info(
        "The selected area or simulation settings changed. Run the simulated "
        "position check again to update the candidate."
    )

if result is None or result["key"] != scenario_key:
    sample_center = _coordinates_at(
        ref_item,
        ref_item["patch_size"] / 2,
        ref_item["patch_size"] / 2,
    )
    (
        preview_image,
        preview_bounds,
        has_full_dem,
        map_reader_error,
    ) = _load_offline_terrain_map(
        ref_item["source_dem"],
        ref_item["file"],
        tuple(ref_item["transform"]),
    )
    st.subheader("Selected area preview")
    st.markdown("**Terrain view**")
    preview_figure = _offline_preview_figure(
        preview_image,
        preview_bounds,
        sample_center,
    )
    st.pyplot(
        preview_figure,
        clear_figure=True,
        width="stretch",
        alt="Terrain preview of the selected test area",
    )
    plt.close(preview_figure)
    if not has_full_dem:
        st.warning(
            "The full elevation map is unavailable here. The terrain view "
            "shows the real selected terrain sample instead."
        )
    st.markdown("**Street map**")
    if sample_center is not None:
        st.pydeck_chart(
            _standard_map(sample_center),
            width="stretch",
            height=480,
        )
        st.markdown(
            "[© OpenStreetMap contributors](https://www.openstreetmap.org/copyright)"
            " · [© CARTO](https://carto.com/attribution/)"
        )
    else:
        st.info("Coordinates are unavailable for this selected terrain sample.")
    st.caption(
        "This preview shows the selected terrain sample, not a position "
        "estimate. Run the simulated position check to see a candidate. "
        "The street map needs an internet connection."
    )

with st.expander("Research diagnostics"):
    if map_reader_error:
        st.markdown("**Terrain map loading detail**")
        st.code(map_reader_error)
    st.markdown("**Available map coverage**")
    st.write(
        f"{len(manifest):,} terrain windows across {len(regions)} mapped regions: "
        f"{', '.join(_display_region(region) for region in regions)}."
    )
    st.markdown("**Positioning method**")
    st.write(
        f"The CNN ranks map windows within {_display_region(selected_region)}. "
        "NCC provides a secondary comparison using the selected terrain sample."
    )
    st.write("The map search uses a 32 pixel grid between candidate windows.")
    st.write(
        "Similarity scores are ranking values, not probabilities. The model "
        "does not provide a calibrated uncertainty estimate."
    )
    if result is not None and result["key"] == scenario_key:
        st.markdown("**Current candidate ranking**")
        candidates = [
            {
                "Rank": candidate.rank,
                "Area": _display_region(candidate.region),
                "Similarity": f"{candidate.similarity:.3f}",
                "Latitude": f"{candidate.latitude:.5f}°" if candidate.latitude is not None else "—",
                "Longitude": f"{candidate.longitude:.5f}°" if candidate.longitude is not None else "—",
            }
            for candidate in result["ranked"]
        ]
        st.dataframe(
            pd.DataFrame(candidates),
            hide_index=True,
            width="stretch",
        )
        st.write(f"CNN search time: {result['cnn_elapsed_ms']:.0f} ms.")

        ncc = result["ncc_result"]
        st.markdown("**NCC cross check**")
        st.write(
            f"Best window within the selected test region: row {ncc.row}, "
            f"column {ncc.col}. NCC score {ncc.score:.3f}; "
            f"search time {result['ncc_elapsed_ms']:.0f} ms."
        )

    if EVALUATION_SUMMARY_PATH.exists():
        evaluation = json.loads(
            EVALUATION_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        cnn_metrics = evaluation.get("cnn", {})
        ncc_metrics = evaluation.get("ncc", {})
        st.markdown("**Saved evaluation results**")
        st.caption(evaluation.get("scope", "Evaluation scope not recorded."))
        eval_cols = st.columns(3)
        if "top1_accuracy" in cnn_metrics:
            eval_cols[0].metric("CNN top 1", f"{cnn_metrics['top1_accuracy']:.1%}")
        if "top3_accuracy" in cnn_metrics:
            eval_cols[1].metric("CNN top 3", f"{cnn_metrics['top3_accuracy']:.1%}")
        if "exact_match_rate" in ncc_metrics:
            eval_cols[2].metric(
                "NCC exact offset",
                f"{ncc_metrics['exact_match_rate']:.1%}",
            )
        st.caption(
            f"Earlier CNN evaluation: {cnn_metrics.get('n_queries', '—')} "
            f"queries on {cnn_metrics.get('n_test_patches', '—')} held out "
            "patches. NCC used synthetic translations."
        )

st.markdown("---")
st.caption(
    "Research prototype using simulated terrain readings and available DEM "
    "coverage. Not validated for real world navigation."
)
