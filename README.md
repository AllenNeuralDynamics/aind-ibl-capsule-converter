# aind-ibl-capsule-converter

This capsule runs the ibl conversion with a manifest file. The manifest is formatted as follows: 

| index | mouseid | sorted_recording | surface_finding | probe_file | probe_name | probe_shank | probe_id | 
|-------|---------|------------------|------------|------------|-------------|----------|------------------
|       |         |                  |            |            |             |          |                  

Column descriptions are below:

| Column | Description |
|--------|-------------|
| # | Row index |
| mouseid | Unique identifier for the mouse subject |
| sorted_recording | Name of the sorted electrophysiology recording session, encoding the mouse ID, recording date/time, and spike-sorting date/time |
| surface_finding | The data asset ID of a separate surface finding recording, if one exists (empty here, indicating no separate surface finding asset) |
| probe_file | Name of the Neuroglancer file used for probe tracing |
| probe_name | Identifier for the specific probe inserted |
| probe_shank | The shank number on a multi-shank probe (empty here, suggesting single-shank or unspecified) |
| probe_id | Identifier linking to a specific id traced in the Neuroglancer JSON file |

An example manifest is below:

| index | mouseid | sorted_recording | surface_finding | probe_file | probe_name | probe_shank | probe_id |
|---|---------|-----------------|-----------------|------------|------------|-------------|----------|
| 0 | 781370 | ecephys_781370_2025-05-30_15-52-48_sorted_2025-10-01_07-00-38 | | 781370_Neuroglancer_GL_2025_09_24 | 45883-1 | | 640-1 |
| 1 | 781370 | ecephys_781370_2025-05-30_15-52-48_sorted_2025-10-01_07-00-38 | | 781370_Neuroglancer_GL_2025_09_24 | 45883-2 | | 640-2 |

The output is an asset in Code Ocean that has the IBL transformed data. See asset here for more details: 
https://codeocean.allenneuraldynamics.org/data-assets/33c8ae97-bb8c-423b-b4b1-3c551224c304/SmartSPIM_776293_2025-03-04_11-24-35_ibl-converted_2025-04-25_13-11-23

### Running capsule
Duplicate the capsule. Then, manually attach the necessary assets below for the sessions in the manifest:
* Primary ecephys asset for each session
* Sorted ecephys asset for each session
* Primary SmartSPIM asset for subject
* Stitched SmartSPIM asset for subject
* Upload manifest to data folder
* Upload neuroglancer json to data folder

Then, in the app panel (shown) below, set the fields for manifest and neuroglancer json by pasting the path for each respective one. 

<img width="656" height="269" alt="image" src="https://github.com/user-attachments/assets/7f09466e-8cf5-4de0-9d91-1d1500f0231a" />

After it runs, you will have to manually create an asset. When creating the asset make sure to do it at the level of the `Run with Parameters...`. 


