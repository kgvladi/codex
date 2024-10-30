# codex
Extracting characteristic features for hi-res images of cancer tissues
ln_seg_pat.py:
  -> reads the original CODEX data file with tiff hi-res images of the biomarkers response
  -> takes a small region in (x,y) coordinates and performs cell segmentation
  -> for each marker makes a new vector with maximal value of the response in each cell
lymph_node_seg_pat.py:
  -> reads the original data file
  -> takes a small region in (x,y) coordinates and performs cell segmentation
  -> replaces the value of the response within each cell with maximal value of the response in each cell
ln_svd.py:
  -> performs svd for the dataset: computes lambdas and constraints G's
ln_backt.py:
  ->filters the outliers in the constraints G's
  ->transforms vectors of the constraints in cell index back to (x,y) basis
ln_svd_calcfit.py:
  -> computes the fit for the data using finite set of constraints for each marker
plot.py
  -> plots the fits
plot_constraints.py
  -> plots the constraints
