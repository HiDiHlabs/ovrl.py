User Guides
===========

Selecting the Number of Principal Components
--------------------------------------------

The selection of an appropriate number of principal components (PCs) can impact the detection of regions with low vertical signal integrity.

While following guidelines may not generalize to all circumstances
(and importantly the selection of the number of PCs is an unsolved problem even in single-cell analysis),
but are meant to help the users to identify good starting points.

- We believe that the number of PCs should be tuned to match the cell-type granularity for the analysis.
- General methods: As is commonly mentioned, methods such as the “elbow” criterion can be helpful in selecting the number of PCs.
  We believe that recommendations made for scRNA seq data will at least partially apply,
  and as such following best practices is advised and can be a helpful guide. For example see https://doi.org/10.15252/msb.20188746.
- Biological expectations: In certain tissue contexts researchers may already have an understanding of overlap signals
  that may arise from the cell type/transcriptional heterogeneity of the tissue or smaller subregions,
  e.g., oligodendrocyte sheaths around neruons in the brain, and apical-basal expression patterns in epithelial cells in the intestine.
  These can be used to "calibrate" the number of PCs and VSI threshold based on whether too few or too many low VSI regions are identified compared to this prior knowledge.
- Exploratory data analysis is an inherently iterative process.
  As such revisiting quality control and the output of tools such as ovrlpy may be triggered by insights from downstream analysis.
  For example, newly identified cell types or unexpected gene expression patterns should be validated to exclude the potential of confounding through cell segmentation artifacts.


What do I do with low-quality (low VSI) cells?
----------------------------------------------

Once regions or cells with low VSI, indicative of potential spatial vertical doublets, have been identified, we recommend that users consider a range of downstream strategies, tailored to the biological question and analytical goals.

For quality control and robust interpretation, one straightforward approach is to annotate low VSI regions or cells to see if they are sensitive to downstream analyses tasks, such as clustering, marker gene identification, and annotation of cell types.
This helps identifying the extent to which low VSI regions/cells confound downstream analysis, thus allowing users to decide to which extent they should be removed.

Alternatively, the presence of low VSI regions can inform deconvolution-based methods such as RCTD or other algorithms that assign multiple cell-type proportions per spatial location.
Users may flag low VSI areas for doublet inference or model these sites as mixtures, especially when seeking to delineate spatially co-located or interacting cell types.

Furthermore, one important consideration is that if low VSI regions are analysed differently (e.g., through removal or deconvolution approaches) this may introduce further confounding factors as low VSI regions and cellular overlaps may not be distributed homogeneously across the sample.

Beyond cell-type annotation, low VSI signals have implications for interpretation in a number of tasks.
Low VSI regions affect the embedding in UMAP analysis, as well as clustering and cell type analysis.
Further, we believe it is reasonable to assume that other analyses such as co-localization analysis, identification of spatial gene programs, or cell-cell communication modeling would also be affected.
For these applications, we recommend explicitly reporting the fraction of low VSI regions and providing sensitivity analyses showing results with and without these regions.
