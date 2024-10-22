Installation
============


To install the necessary tools and dependencies for this project, follow the steps outlined below. These instructions will guide you through setting up the environment for both standard use and interactive analysis with Jupyter notebooks.

.. note::
   Make sure you have Python and pip installed on your machine before proceeding.

Steps for Installation
-----------------------

1. **Clone the Repository**

   First, ensure that you have cloned the repository to your local machine. If you haven't already done so, use the following commands:

   .. code-block:: bash

      git clone https://github.com/HiDiHlabs/ovrl.py.git
      cd ovrl.py


2. **Install the Package in Editable Mode**

   To install the ovrlpy package, execute the following command:

   .. code-block:: bash

      pip install .

   This installs the package based on the current state of the source files.

3. **Set Up for Interactive Analysis (Optional)**

   If you plan to use Jupyter notebooks for interactive analysis or the project's tutorials, you'll need to install some additional packages: **Jupyter**, **pyarrow**, and **fastparquet**. Install them using:

   .. code-block:: bash

      pip install jupyter pyarrow fastparquet


Summary of Commands
-------------------

Here's a summary of the commands to run for installation:

.. code-block:: bash

   # Step 1: Clone the repository
   git clone https://github.com/HiDiHlabs/ovrl.py.git
   cd ovrl.py

   # Step 2: Install package in editable mode
   pip install .

   # Step 3: Install Jupyter and other packages for interactive analysis
   pip install jupyter pyarrow fastparquet

