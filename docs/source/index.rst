Ovrlpy
======

**ovrlpy** is a python tool to investigate cell overlaps in imaging-based spatial transcriptomics data.

Introduction
------------

In spatial biology, tissue slices are commonly used to study the spatial distribution of cells and molecules. However, since these slices represent 3D structures in 2D, overlapping structures in the vertical dimension can lead to artefacts and inconsistencies in the data.

**ovrlpy** is a quality-control tool for spatial transcriptomics data that can help analysts find sources of vertical signal inconsistency in their data.
It is works with imaging-based spatial transcriptomics data, such as 10x Genomics' Xenium or Vizgen's MERSCOPE platforms.
The main feature of the tool is the production of 'signal integrity maps' that can help analysts identify sources of signal inconsistency in their data.
Users can also use the built-in 3D visualisation tool to explore regions of signal inconsistency in their data on a molecular level.


.. image:: ../resources/cell_overlap_visualization.jpg
   :alt: 3D slice visualization
   :align: center
   :width: 600px

Citation
--------

If you are using `ovrlpy` for your research please cite

Tiesmeyer, S., Müller-Bötticher, N., Malt, A., Ma, L., Marco-Salas, S., Kiessling, P., ... & Ishaque, N.
(2026).
Identifying 3D signal overlaps in spatial transcriptomics data with ovrlpy.
*Nature Biotechnology*, 1-5. https://doi.org/10.1038/s41587-026-03004-8


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   self
   installation
   usage
   tutorials/index
   guide
   how_to/index


Indices and tables
==================

-  :ref:`genindex`
-  :ref:`search`
