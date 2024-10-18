
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

The tool can be installed using the requirements.txt file in the root directory of the repository.

```bash
pip install -e .
```

In order to use the ipython notebooks and perform interactive analysis, you will need to install the jupyter package also. For the tutorials, pyarrow and fastparquet are also required.

```bash
pip install jupyter pyarrow fastparquet
```

## quickstart

The simplest use case of ovrlpy is the creation of a signal integrity map from a spatial transcriptomics dataset.
In a first step, we define a number of parameters for the analysis:

```python
import pandas as pd
import ovrlpy

# define ovrlpy analysis parameters:
kde_bandwidth = 2
n_expected_celltypes=20

# load the data

coordinate_df = pd.read_csv('path/to/coordinate_file.csv')
coordinate_df.head()
```

you can then fit an ovrlpy model to the data and create a signal integrity map:

```python

# fit the ovrlpy model to the data

from ovrlpy import ovrlp 

integrity, signal, visualizer = ovrlp.compute_coherence_map(df=coordinate_df,KDE_bandwidth=kde_bandwidth,n_expected_celltypes=n_expected_celltypes)

```

returns a signal integrity map, a signal map and a visualizer object that can be used to visualize the data:

```python
visualizer.plot_fit()
```

and visualize the signal integrity map:

```python
fig, ax = ovrlp.plot_signal_integrity(integrity,signal,signal_threshold=4.0)
```

Ovrlpy can also identify individual overlap events in the data:

```python
import matplotlib.pyplot as plt
doublet_df = ovrlp.detect_doublets(integrity,signal,signal_cutoff=4,coherence_sigma=1)

doublet_df.head()
```

And use the visualizer to show a 3D visualization of the overlaps in the tissue:

```python
window_size=60          # size of the window around the doublet to show
n_doublet_to_show = 0   # index of the doublet to show
x,y = doublet_df.loc[doublet_case,['x','y']] # location of the doublet event

# subsample the data around the doublet event
subsample = visualizer.subsample_df(x,y,coordinate_df,window_size=window_size)  
# transform the subsample using the fitted color embedding model
subsample_embedding, subsample_embedding_color = visualizer.transform(subsample)

# plot the subsample instance:
visualizer.plot_instance(subsample,subsample[['x','y']].values,subsample_embedding_color,x,y,window_size=window_size)

```
