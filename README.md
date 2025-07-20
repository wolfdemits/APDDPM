# Vakoverschrijdend project
Development and Validation of a Dose-Aware Anchored Path Diffusion Denoising Probabilistic Model for Low-Dose PET
Imaging. 

## Used file structure:
<pre>
|_ DATA
    |_ DICOM
        |_ HIGH RES
            |_ ...
        |_ LOW RES
            |_ ...
    |_ ZARR_PREPROCESSED
        |_ ... (same as DICOM)
    |_ ZARR_RESIDUALS
        |_ ... (same as DICOM)
   (|_ BLACKLIST.json)
    |_ Residual_stats.json
|_ PRE-PROCESSING
    |_ visualizer.py
    |_ Slicemanager.py
    |_ preprocessing.py
    |_ residual-preprocessing.py
|_ ML-MODEL
    |_ ... 
|_ Testing
</pre>

Zarr array file structure: <br>
Resolution -> location -> mouse -> plane -> index (same as DICOM structure)

**Use Visualizer.py to inspect and remove slices**

## Used slice ID format:
ID: MMLPXXR <br>
-> MM: Patient ID <br>
-> L: Location     H = HEAD-THORAX; T = THORAX-ABDOMEN <- ? <br>
-> P: Plane        C = Coronal; S = Sagittal; T = Transax <- ? <br>
-> XX: Slice ID <br>
-> R: Resoulution  H = HIGH RES; L = LOW RES <br>
<br>
Example: 01HC01L = 1st low-resolution slice of coronal Head-Thorax image of patient 1<br>
<br>
BLACKLIST.json contains IDs in format: MMLPXX (where resolution parameter is dropped (because both get blacklisted))

## Data
**Data/ directory on onedrive**:
https://ugentbe-my.sharepoint.com/..............
