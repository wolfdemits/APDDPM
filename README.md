# Vakoverschrijdend project
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
|_ PRE-PROCESSING
    |_ datamanager.py
    |_ convert-to-zarr.py
|_ ML-MODEL
    |_ ... 
|_ Testing
    |_ visualizer.py (uses ./PRE-PROCESSING/datamanager.py)
</pre>

data.json: -> contains info about tracers and patient IDs, used by filemanager <br>
info.json: -> stores info about specific acquisition of patient. I.e.: WB/BR study, tracer, voxel size, time point, TOF resolution, ... 

**Use Visualizer.py to inspect images**