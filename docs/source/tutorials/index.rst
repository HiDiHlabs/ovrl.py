Tutorials 
==========================

We will demonstrate an example usage of ovrlpy on 3 different datasets (`Xenium Brain <https://www.10xgenomics.com/products/xenium-in-situ/mouse-brain-dataset-explorer>`_,
`Vizgen liver <https://info.vizgen.com/mouse-liver-data>`_, `Vizgen receptor <https://info.vizgen.com/mouse-brain-map>`_  ). 


Installation
""""""""""""
1. **Clone the Repository**

   First, ensure that you have cloned the repository to your local machine. If you haven't already done so, use the following commands:

   .. code-block:: bash

      git clone https://github.com/HiDiHlabs/ovrl.py.git
      cd ovrl.py


2. **Install the Tutorial of the package**

   To install the tutorial for this project, execute the following command:

   .. code-block:: bash

      pip install ovrlpy[tutorial] 


   This will install the required dependencies and tutorial-specific components of the package.

3. **Start with the Tutorials**
   To start the tutorial JupyterNotebooks are stored in 
   .. code-block:: bash

      ovrl.py/docs/source/tutorials/*.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   xenium_brain
   vizgen_liver
   vizgen_receptor
