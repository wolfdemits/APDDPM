# APPDPM
Development and Validation of a Dose-Aware Anchored Path Diffusion Denoising Probabilistic Model for Low-Dose PET
Imaging. 

## Used file structure:
<pre>
|_ DATA
    |_ PATIENTS
      |_ [ID]
      |_ info.json
    |_ data.json

    (|_ RESIDUALS)
    (|_ Residual_stats.json)
|_ PREPROCESSING
    |_ datamanager.py
    |_ convert-to-zarr.py
|_ ML_MODEL
    |_ ... 
|_ VISUALIZATION
    |_ visualizer.py (uses ./PREPROCESSING/datamanager.py)
    |_ plotting_functions.py
    |_ plotting_script.py
|_ ANALYSIS
    |_ sphericalVOI_analysis.py
</pre>

data.json: -> contains info about tracers and patient IDs, used by filemanager <br>
info.json: -> stores info about specific acquisition of patient. I.e.: WB/BR study, tracer, voxel size (blb 4 = 2mm), time point, TOF resolution, array shape, divisions and reconstruction iterations 

**Use Visualizer.py to inspect images**