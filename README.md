# codex
Extracting characteristic features for hi-res images of cancer tissues
<br/> **ln_seg_pat.py**:
<br/>   -> reads the original CODEX data file with tiff hi-res images of the biomarkers response
<br/>   -> takes a small region in (x,y) coordinates and performs cell segmentation
<br/>   -> for each marker makes a new vector with maximal value of the response in each cell
<br/>**lymph_node_seg_pat.py**:
<br/>   -> reads the original data file
<br/>   -> takes a small region in (x,y) coordinates and performs cell segmentation
<br/>   -> replaces the value of the response within each cell with maximal value of the response in each cell
<br/>**ln_svd.py**:
<br/>   -> performs svd for the dataset: computes lambdas and constraints G's
<br/>**ln_backt.py**:
<br/>   ->filters the outliers in the constraints G's
<br/>   ->transforms vectors of the constraints in cell index back to (x,y) basis
<br/>**ln_svd_calcfit.py**:
<br/>   -> computes the fit for the data using finite set of constraints for each marker
<br/>**plot.py**
<br/>   -> plots the fits
<br/>**plot_constraints.py**
<br/>   -> plots the constraints
