# Grid coordinates and offsets

Terrain grids are indexed as `terrain[row][column]`.

- Row index increases downward in the grid (south on a conventional north-up map).
- Column index increases to the right (east on a conventional north-up map).
- A source cell at `(row, column)` translated by `(row_offset, col_offset)` appears at `(row + row_offset, column + col_offset)`.
- Positive row offsets therefore move terrain down/south; positive column offsets move it right/east. Negative values move in the opposite direction.
- Offsets are integer grid cells (pixels), not metres or geographic coordinates.

`shift_terrain` moves cells without wrapping and fills exposed cells with a
constant. `normalized_cross_correlation` searches candidate translations and
compares only overlapping cells. Its returned offset uses the same convention:
it is the translation that maps reference cells into the observation grid.
The NCC score is Pearson correlation on that overlap, ranging from -1 to 1.

This convention describes array coordinates only. Converting offsets to real
geographic displacement requires the GeoTIFF's pixel size, orientation, and
coordinate reference system; that conversion is outside this core module.
