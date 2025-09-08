# APPDPM
Development and Validation of a Dose-Aware Anchored Path Denoising Diffusion Probabilistic Model for Low-Dose PET
Imaging. 

## Used file structure:
<pre>
|_ DATA
    |_ PATIENTS
      |_ [ID]
            |_ div[x]
        |_ info.json
    |_ data.json
|_ DATA_PREPROCESSED
    |_ PATIENTS
        |_ [ID]
            |_ [PLANE]
                |_ div[x]
                    |_ [idx]
            |_ info.json
    |_ data.json
|_ PREPROCESSING
    |_ datamanager.py
    |_ convert-to-zarr.py
    |_ residual_RMS_stats.py
|_ ML_MODEL
    |_ APD.py
    |_ APD_test_animation.py
    |_ UNet.py
    |_ training.py
    |_ helperfunctions.py
    |_ embedding.py
|_ VISUALIZATION
    |_ visualizer.py (uses ./PREPROCESSING/datamanager.py)
    |_ plotting_functions.py
    |_ plotting_script.py
|_ ANALYSIS
    |_ sphericalVOI_analysis.py
    |_ circularROI_analysis.py
|_ RESIDUALS
    |_ PATIENTS
        |_ [ID]
            |_ [PLANE]
                |_ div[x]
                    |_ [idx]
            |_ info.json
    |_ data.json
    |_ residualstats.json???
|_ RESULTS
    |_ CHECKPOINTS
    |_ FIGURES
|_ datasets.json
</pre>

data.json: -> contains info about tracers and patient IDs, used by filemanager <br>
info.json: -> stores info about specific acquisition of patient. I.e.: WB/BR study, tracer, voxel size (blb 4 = 2mm), time point, TOF resolution, array shape, divisions and reconstruction iterations 

**Use Visualizer.py to inspect images**