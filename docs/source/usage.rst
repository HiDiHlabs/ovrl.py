Usage
=====

Quickstart
----------

This quickstart guide will walk you through the basic steps of using **ovrlpy** to create a signal integrity map from a imaging-based spatial transcriptomics dataset. Follow the steps below to get started.

1. Set Up Parameters and Load Your Data
_______________________________________

Start by defining the key parameters for the analysis and loading your spatial transcriptomics data.

.. code-block:: python

   import pandas as pd
   import ovrlpy

   # Define analysis parameters for ovrlpy
   kde_bandwidth = 2  # The smoothness of the kernel density estimation (KDE)
   n_expected_celltypes = 20  # Number of expected cell types in the data

   # Load your spatial transcriptomics data from a CSV file
   coordinate_df = pd.read_csv('path/to/coordinate_file.csv')

   # Display the first few rows of the dataset
   coordinate_df.head()

In this step, we load the dataset and configure the model parameters, such as `kde_bandwidth` (to control smoothness) and `n_expected_celltypes` (to set the expected number of cell types).

2. Fit the ovrlpy Model
_______________________

Fit the **ovrlpy** model to generate the signal integrity map.

.. code-block:: python

   from ovrlpy import ovrlp

   # Fit the ovrlpy model to the spatial data
   integrity, signal, visualizer = ovrlp.compute_coherence_map(
       df=coordinate_df,
       KDE_bandwidth=kde_bandwidth,
       n_expected_celltypes=n_expected_celltypes
   )

This function generates:
- **integrity**: The signal integrity map.
- **signal**: The signal map representing the strength of spatial signals.
- **visualizer**: A visualizer object that helps to plot and explore the results.

3. Visualize the Model Fit
__________________________

Once the model is fitted, you can visualize how well it matches your spatial data.

.. code-block:: python

   # Use the visualizer object to plot the fitted signal map
   visualizer.plot_fit()

This plot gives you a visual representation of the models fit to the spatial transcriptomics data.

4. Plot the Signal Integrity Map
________________________________

Now, plot the signal integrity map using a threshold to highlight areas with strong signal coherence.

.. code-block:: python

   # Plot the signal integrity map with a signal threshold
   fig, ax = ovrlp.plot_signal_integrity(integrity, signal, signal_threshold=4.0)


5. Detect and Visualize Overlaps (Doublets)
___________________________________________

Identify overlapping signals (doublets) in the tissue and visualize them.

.. code-block:: python

   import matplotlib.pyplot as plt

   # Detect doublet events (overlapping signals) in the dataset
   doublet_df = ovrlp.detect_doublets(
       integrity,
       signal,
       signal_cutoff=4,  # Threshold for signal strength
       coherence_sigma=1  # Controls the coherence of the signals
   )

   # Display the detected doublets
   doublet_df.head()

6. 3D Visualization of a Doublet Event
______________________________________

Visualize a specific overlap event (doublet) in 3D to see how it looks in the tissue.

.. code-block:: python

   # Parameters for 3D visualization
   window_size = 60  # Size of the visualization window around the doublet
   n_doublet_to_show = 0  # Index of the doublet to visualize

   # Get the coordinates of the doublet event
   x, y = doublet_df.loc[n_doublet_to_show, ['x', 'y']]

   # Subsample the data around the doublet event
   subsample = visualizer.subsample_df(x, y, coordinate_df, window_size=window_size)

   # Transform the subsample using the color embedding model
   subsample_embedding, subsample_embedding_color = visualizer.transform(subsample)

   # Plot the doublet event with 3D visualization
   visualizer.plot_instance(
       subsample,
       subsample[['x', 'y']].values,
       subsample_embedding_color,
       x, y,
       window_size=window_size
   )

This visualization shows a 3D representation of the spatial overlap event, giving more insight into the structure and coherence of the signals.
