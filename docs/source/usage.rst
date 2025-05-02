Quickstart
==========

This quickstart guide will walk you through the basic steps of using **ovrlpy** to create a signal integrity map from a imaging-based spatial transcriptomics dataset. Follow the steps below to get started.

1. Set Up Parameters and Load Your Data
_______________________________________

Start by defining the key parameters for the analysis and loading your spatial transcriptomics data.
The dataset should contain a *x*, *y*, and *z* columns (in um) and a *gene*  column.
Functions to read the data in the correct format are available for common file formats
(such as output from Xenium, Vizgen, and CosMx).

.. code-block:: python

   import pandas as pd
   import ovrlpy

   # Define analysis parameters for ovrlpy
   kde_bandwidth = 2.5  # smoothness of the kernel density estimation (KDE)
   n_components = 20  # number of principal components, depends on the data complexity

   # Load your spatial transcriptomics data from a CSV file
   coordinate_df = pd.read_csv('path/to/coordinate_file.csv')


In this step, we load the dataset and configure the model parameters, such as
`kde_bandwidth` (to control smoothness) and
`n_components` (to set the number of prinicpal components that will be used).

2. Fit the ovrlpy Model
_______________________

Fit the **ovrlpy** model to generate the signal integrity map.

.. code-block:: python

   # Fit the ovrlpy model to the spatial data
   dataset = ovrlpy.Ovrlp(
       coordinate_df,
       KDE_bandwidth=kde_bandwidth,
       n_components=n_components,
       n_workers=4,  # number of threads to use for processing
   )
   dataset.analyse()

3. Visualize the Model Fit
__________________________

Once the model is fitted, you can visualize how well it matches your spatial data.

.. code-block:: python

   fig = ovrlpy.plot_pseudocells(dataset)

This plot gives you a visual representation of the models fit to the spatial transcriptomics data.

4. Plot the Signal Integrity Map
________________________________

Now, plot the signal integrity map using a threshold to highlight areas with strong signal coherence.

.. code-block:: python

   fig = ovrlpy.plot_signal_integrity(dataset, signal_threshold=4)


5. Detect and Visualize Overlaps (Doublets)
___________________________________________

Identify overlapping signals (doublets) in the tissue and visualize them.

.. code-block:: python

   # Detect doublet events (overlapping signals) in the dataset
   doublets = dataset.detect_doublets(
       min_signal=4,  # threshold for signal strength
       integrity_sigma=1,  # controls the coherence of the signals
   )

   doublets.head()

6. 3D Visualization of a Doublet Event
______________________________________

Visualize a specific overlap event (doublet) to see how it looks in the tissue.

.. code-block:: python

   # Parameters for the visualization
   window_size = 60  # Size of the visualization window around the doublet
   doublet_to_show = 0  # Index of the doublet to visualize

   # Coordinates of the doublet event
   x, y = doublets["x", "y"].row(doublet_to_show)

   # Plot the doublet event with 3D visualization
   fig = ovrlpy.plot_region_of_interest(dataset, x, y, window_size=window_size)

This visualization shows a top/bottom/side representation of the spatial overlap event,
giving more insight into the structure and coherence of the signals.
