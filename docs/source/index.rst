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

Tiesmeyer, S., Müller-Bötticher, N., Malt, A., Long, B., Marco-Salas, S., Kiessling, P., ... & Ishaque, N. (2025).
2D, or not 2D? Investigating Vertical Signal Integrity of Tissue Slices.
*bioRxiv* https://doi.org/10.1101/2025.01.13.632601


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   self
   installation
   usage
   tutorials/index


Indices and tables
==================

-  :ref:`genindex`
-  :ref:`search`
