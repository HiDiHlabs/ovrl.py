
<!-- include image 'documentation/resources/ovrlpy-logo.png -->
![ovrlpy logo](docs/resources/ovrlpy-logo.png)

A python tool to investigate vertical signal properties of imaging-based spatial transcriptomics data.

## introduction

Much of spatial biology uses microscopic tissue slices to study the spatial distribution of cells and molecules. In the process, tissue slices are often interpreted as 2D representations of 3D biological structures - which can introduce artefacts and inconsistencies in the data whenever structures overlap in the thin vertical dimension of the slice:

![3D slice visualization](docs/resources/cell_overlap_visualization.jpg)

Ovrl.py is a quality-control tool for spatial transcriptomics data that can help analysts find sources of vertical signal inconsistency in their data.
It is works with imaging-based spatial transcriptomics data, such as 10x genomics' Xenium or vizgen's MERSCOPE platforms.
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
n_components = 20 # number pf PCA components

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

fig = ovrlpy.plot_region_of_interest(dataset, x, y, window_size=50)
```

![plot_region_of_interest output](docs/resources/plot_roi.png)

### parameter selection
[![](https://mermaid.ink/img/pako:eNp9lG9P2zAQxr_Kybwgldqq_1uiiam0wAobQgMhbYQXTuyk3hI7sh1Y1_a77-KEkg1pzZv6_LvnnrvY2ZJIMU580ul0AhkpGYvEDySAXfOM-3AcUsOPD4EHqgUNU24cAxDS6GeiVSEZokex-yHtxAIZp-olWlNt4X5Z8XPv8bOiDO5yagVN4V5TaSItcqsyERlg1GI9-9SCTucUzryArKRwZE41zbjl2nwI9anHu0kXboos5BpUDLdayEjkyC1UlivJpTWtgLSqqqYIE03zNahnneabKnjmSiy8BU2jIqWWw8Pdqk5YuL2ld_5M06LeAiaM1SIsrFCy5paOO0ebd9w6yK41N2uVMucS-0d_loq0NPm3wKu5c-h0UWQnYsiVMQKnu4MLlHygqWBlcaYKDNpG44URMoHl_Hb1TueymWmFMQWHXPNyfGVdwLchYhpZAx5WpHIDDKcaWc7epC4qqU_br9wUKaIGc02ZpfTm476CLmuoWq2qaVYLLtk_g2fqRWLznGYHs4hfeXMplW32CLHSDRoN0nRjhKmdXbm8a-9x-R55qplPjtl942YH11Xoup7xCs9PWQ07lzzixlC92cGq4fo_B6bWvVGY4c3Zj8LYer9xOFsHLdImiRaM-DFNDW-TjOuMlmuyLZmAuAsVEB__lpcsIO1G_HDRSmBbiZbc63Wr8o7iUfkEpATwveyxaE7ld6Uy4ltdYFmkk_XBRJGX52IpKLb3hqBbrhcoa4k_6A37ToT4W_KL-MPBtDsYnkzGo_Go3-uPJ22yIf542j2ZDEazwWiMm4PxbN8mv13VXnc66Q2HJ_3RpDed9WZjVONM4Mn5Un1o3Pdm_wdW4Wnz?type=svg)](https://mermaid.live/edit#pako:eNp9lG9P2zAQxr_Kybwgldqq_1uiiam0wAobQgMhbYQXTuyk3hI7sh1Y1_a77-KEkg1pzZv6_LvnnrvY2ZJIMU580ul0AhkpGYvEDySAXfOM-3AcUsOPD4EHqgUNU24cAxDS6GeiVSEZokex-yHtxAIZp-olWlNt4X5Z8XPv8bOiDO5yagVN4V5TaSItcqsyERlg1GI9-9SCTucUzryArKRwZE41zbjl2nwI9anHu0kXboos5BpUDLdayEjkyC1UlivJpTWtgLSqqqYIE03zNahnneabKnjmSiy8BU2jIqWWw8Pdqk5YuL2ld_5M06LeAiaM1SIsrFCy5paOO0ebd9w6yK41N2uVMucS-0d_loq0NPm3wKu5c-h0UWQnYsiVMQKnu4MLlHygqWBlcaYKDNpG44URMoHl_Hb1TueymWmFMQWHXPNyfGVdwLchYhpZAx5WpHIDDKcaWc7epC4qqU_br9wUKaIGc02ZpfTm476CLmuoWq2qaVYLLtk_g2fqRWLznGYHs4hfeXMplW32CLHSDRoN0nRjhKmdXbm8a-9x-R55qplPjtl942YH11Xoup7xCs9PWQ07lzzixlC92cGq4fo_B6bWvVGY4c3Zj8LYer9xOFsHLdImiRaM-DFNDW-TjOuMlmuyLZmAuAsVEB__lpcsIO1G_HDRSmBbiZbc63Wr8o7iUfkEpATwveyxaE7ld6Uy4ltdYFmkk_XBRJGX52IpKLb3hqBbrhcoa4k_6A37ToT4W_KL-MPBtDsYnkzGo_Go3-uPJ22yIf542j2ZDEazwWiMm4PxbN8mv13VXnc66Q2HJ_3RpDed9WZjVONM4Mn5Un1o3Pdm_wdW4Wnz)
