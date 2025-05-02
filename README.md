
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

`ovrlpy` can be installed from [PyPI](https://pypi.org/project/ovrlpy/) or
[bioconda](https://bioconda.github.io/recipes/ovrlpy/README.html)

```bash
# install from PyPI
pip install ovrlpy

# or install from bioconda
conda install bioconda::ovrlpy
```

## quickstart

The simplest use case of ovrlpy is the creation of a signal integrity map from a spatial transcriptomics dataset.
In a first step, we define a number of parameters for the analysis:

```python
import pandas as pd
import ovrlpy

# define ovrlpy analysis parameters
n_components = 20

# load the data
coordinate_df = pd.read_csv('path/to/coordinate_file.csv')
coordinate_df.head()
```

the coordinate dataframe should contain a *gene*, *x*, *y*, and *z* column.

you can then fit an ovrlpy model to the data and create a signal integrity map:

```python
# fit the ovrlpy model to the data
dataset = ovrlpy.Ovrlp(
    coordinate_df,
    n_components=n_components,
    n_workers=4,  # number of threads to use for processing
)

dataset.analyse()
```

after fitting we can visualize the data ...

```python
fig = ovrlpy.plot_pseudocells(dataset)
```
![plot_fit output](docs/resources/plot_fit.png)


... and the signal integrity map

```python
fig = ovrlpy.plot_signal_integrity(dataset, signal_threshold=4)
```

![plot_signal_integrity output](docs/resources/xenium_integrity_with_highlights.svg)

Ovrlpy can also identify individual overlap events in the data:

```python
doublets = dataset.detect_doublets(min_signal=4, integrity_sigma=1)
```

And plot a multi-view visualization of the overlaps in the tissue:

```python
# Which doublet do you want to visualize?
doublet_to_show = 0

x, y = doublets["x", "y"].row(doublet_to_show)

fig = ovrlpy.plot_region_of_interest(dataset, x, y, window_size=window_size)
```

![plot_region_of_interest output](docs/resources/plot_roi.png)
