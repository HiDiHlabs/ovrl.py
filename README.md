
<!-- include image 'documentation/resources/ovrlpy-logo.png -->
![ovrlpy logo](docs/resources/ovrlpy-logo.png)

A python tool to investigate vertical signal properties of imaging-based spatial transcriptomics data.

## introduction

Much of spatial biology uses microscopic tissue slices to study the spatial distribution of cells and molecules. In the process, tissue slices are often interpreted as 2D representations of 3D biological structures - which can introduce artefacts and inconsistencies in the data whenever structures overlap in the thin vertical dimension of the slice:

![3D slice visualization](docs/resources/cell_overlap_visualization.jpg)

Ovrl.py is a quality-control tool for spatial transcriptomics data that can help analysts find sources of vertical signal inconsistency in their data.
It is works with imaging-based spatial transcriptomics data, such as 10x genomics' Xenium or vizgen's MERFISH platforms.
The main feature of the tool is the production of 'signal integrity maps' that can help analysts identify sources of signal inconsistency in their data.
Users can also use the built-in 3D visualisation tool to explore regions of signal inconsistency in their data on a molecular level.

## installation

`ovrlpy` can be installed from [PyPI](https://pypi.org)

```bash
pip install ovrlpy
```

## quickstart

The simplest use case of ovrlpy is the creation of a signal integrity map from a spatial transcriptomics dataset.
In a first step, we define a number of parameters for the analysis:

```python
import pandas as pd
import ovrlpy

# define ovrlpy analysis parameters:
n_expected_celltypes = 20

# load the data
coordinate_df = pd.read_csv('path/to/coordinate_file.csv')
coordinate_df.head()
```

the coordinate dataframe should contain a *gene*, *x*, *y*, and *z* column.

you can then fit an ovrlpy model to the data and create a signal integrity map:

```python
# fit the ovrlpy model to the data
signal_integrity, signal_strength, visualizer = ovrlpy.run(
    coordinate_df, n_expected_celltypes=n_expected_celltypes
)
```

returns a signal integrity map, a signal map and a visualizer object that can be used to visualize the data:

```python
visualizer.plot_fit()
```
![plot_fit output](docs/resources/plot_fit.png)


and visualize the signal integrity map:

```python
fig, ax = ovrlpy.plot_signal_integrity(signal_integrity, signal_strength, signal_threshold=4)
```

![plot_signal_integrity output](docs/resources/xenium_integrity_with_highlights.svg)

Ovrlpy can also identify individual overlap events in the data:

```python
doublet_df = ovrlpy.detect_doublets(
    signal_integrity, signal_strength, minimum_signal_strength=3, integrity_sigma=2
)

doublet_df.head()
```

And use the visualizer to show a 3D visualization of the overlaps in the tissue:

```python
# Which doublet do you want to visualize?
n_doublet_case = 0

x, y = doublet_df.loc[doublet_case, ["x", "y"]]

ovrlpy.plot_region_of_interest(
    x,
    y,
    coordinate_df,
    visualizer,
    signal_integrity,
    signal_strength,
)
```

![plot_region_of_interest output](docs/resources/plot_roi.png)
