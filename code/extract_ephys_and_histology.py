from aind_mri_utils.file_io.neuroglancer import read_neuroglancer_annotation_layers
from aind_mri_utils.file_io.neuroglancer import get_image_source, _load_json_file
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os
import shutil
import warnings
import zarr


import ants
import json
from iblatlas import atlas

import argparse

# Ephy readers
from aind_ephys_ibl_gui_conversion.ephys import extract_continuous
from aind_ephys_ibl_gui_conversion.ephys import extract_spikes
# Etc.
from aind_ephys_ibl_gui_conversion.histology import create_slicer_fcsv
from aind_ephys_ibl_gui_conversion.histology import probe_df_to_fcsv
from aind_ephys_ibl_gui_conversion.histology import projected_onto_line
from aind_ephys_ibl_gui_conversion.histology import order_annotation_pts
from aind_ephys_ibl_gui_conversion.histology import read_json_as_dict
from aind_ephys_ibl_gui_conversion.histology import __read_zarr_image
from aind_ephys_ibl_gui_conversion.histology import check_orientation
from aind_ephys_ibl_gui_conversion.histology import get_highest_level_info
from aind_ephys_ibl_gui_conversion.histology import get_additional_channel_image_at_highest_level


# Updated version of https://github.com/AllenNeuralDynamics/aind-mri-utils/blob/3e757c6a58676dac345b936f4215322e0a923494/src/aind_mri_utils/file_io/neuroglancer.py#L296
def get_image_source(filename: str) -> list[str]:
    """
    Reads image source url from a Neuroglancer JSON file.
    
    Returns a list of image sources.    
    """
    data = _load_json_file(filename)

    image_layers = [x for x in data['layers'] if x['type'] == 'image']
    return [layer['source'] if isinstance(layer['source'], str) else layer['source']['url'] for layer in image_layers] 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--manifest',
                    dest = 'annotation_manifest',
                    default = '729293/Manifest_Day1_2_729293 1.csv',
                    help = 'Probe Annotations')
    
    parser.add_argument('--neuroglancer',
                        dest = 'neuroglancer',
                        default = 'Probes_561_729293_Day1and2.json',
                        help = 'Directory containing probe annotations')
                        
    parser.add_argument('--legacy_registration',
                        dest = 'legacy_registration',
                        default =None,
                        help = 'Use old registration from Di capsual')
                        
    args = parser.parse_args()
    if args.legacy_registration == '':
        args.legacy_registration = None
                        

    # Find the neuroglancer file
    if not ('/data/' in args.neuroglancer):
        neuroglancer_file_path = os.path.join('/data/',args.neuroglancer)
    else:
        neuroglancer_file_path = args.neuroglancer
                        
    if not ('/data/' in args.annotation_manifest):
        annotation_manifest_path = os.path.join('/data/',args.annotation_manifest)
    else:
        annotation_manifest_path = args.annotation_manifest
    
    # for debugging, keep a record of the manifest used
    Path('/results/manifest.csv').write_bytes(Path(annotation_manifest_path).read_bytes())

    # Read the annotation-ephys pairings
    manifest_df = pd.read_csv(annotation_manifest_path)

    # Load the template and the ccf
    template = ants.image_read('/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz')
    ccf_25 = ants.image_read('/data/allen_mouse_ccf/average_template/average_template_25.nii.gz')
    ccf_annotation_25 = ants.image_read('/data/allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz')
    brain_atlas = atlas.AllenAtlas(25,hist_path='/scratch/')
    
    # Default is to use the pipeline registration. 
    # However, if an alternative path is passed as "legacy_registration", we will pull the regisration from there.
    # This assumes that you used Di's conversion capsual. If you don't know what that means, you probably didnt...
    if args.legacy_registration == None:
        # Image source will be in a neuroglancer layer.
        # This assumes that a matching stiched asset is attached to the file.
        sources = get_image_source(neuroglancer_file_path)
        if len(sources) > 1:
            print("Found multiple SmartSPIM sources in Neuroglancer file. Using source for first layer.")
        source = sources[0]
        print(f"Using SmartSPIM source: {source}")
        smartspim_session_id = next(x for x in source.split('/') if x.startswith('SmartSPIM_'))
        registration_data_asset = os.path.join('/data/', smartspim_session_id)
        
        # Find data
        alignment_channel = np.sort(os.listdir(os.path.join(registration_data_asset,'image_atlas_alignment')))[-1]
        prep_image_folder = os.path.join(registration_data_asset,'image_atlas_alignment',alignment_channel,'metadata','registration_metadata')
        moved_image_folder = os.path.join(registration_data_asset,'image_atlas_alignment',alignment_channel)

    else:        
        # Handle legacy path
        if not ('/data/' in args.legacy_registration):
            registration_data_asset = os.path.join('/data/',args.legacy_registration)
        else:
            registration_data_asset = args.legacy_registration
        
        prep_image_folder = os.path.join(registration_data_asset,'registration')
        moved_image_folder = os.path.join(registration_data_asset,'registration')

    zarr_read = ants.image_read(os.path.join(prep_image_folder,'prep_n4bias.nii.gz'))
    extrema = np.array(zarr_read.shape)*np.array(zarr_read.spacing)
    offset = zarr_read.origin
    
    # Get CCF space histology for this mouse
    histology_results = os.path.join('/results',str(manifest_df.mouseid[0]),'ccf_space_histology')
    os.makedirs(histology_results,exist_ok = True)
    
    # Read the registration channel data. No need to re-transform since this was done as part of inital registration
    outimg = ants.image_read(os.path.join(moved_image_folder,'moved_ls_to_ccf.nii.gz'))
    ants.image_write(outimg,os.path.join(histology_results,f'histology_registration.nrrd'))
    
    # Handle other channel data. Depending on legacy flag, this may still need to be computed.
    image_histology_results = os.path.join('/results',str(manifest_df.mouseid[0]),'image_space_histology')
    os.makedirs(image_histology_results,exist_ok = True)
    shutil.copy(os.path.join(prep_image_folder,'prep_n4bias.nii.gz'),
                os.path.join(image_histology_results,'histology_registration.nii.gz'))
    
    if args.legacy_registration != None:
        raise NotImplementedError('Legacy Flag has been removed. If this is a problem, it needs to be reimplemented')
        # Handle other channels: This is a work in progress
        other_files = [x for x in os.listdir(moved_image_folder) if 'moved_ls_to_template_' in x and '.nii.gz' in x]
        for fl in other_files:
            chname = fl.split('moved_ls_to_template_')[-1].split('.nii.gz')[0]
            image_in_template = ants.image_read(os.path.join(registration_data_asset,'registration',fl))
            outimg = ants.apply_transforms(ccf_25,image_in_template,['/data/spim_template_to_ccf/syn_1Warp.nii.gz','/data/spim_template_to_ccf/syn_0GenericAffine.mat'])

            ants.image_write(outimg,os.path.join(histology_results,f'histology_{chname}.nrrd'))
            shutil.copy(os.path.join(registration_data_asset,'registration',fl),os.path.join(image_histology_results,'histology_{chname}.nii.gz'))
    else:
        # find channels not used for alignment
        stitched_zarrs=  os.path.join(registration_data_asset,'image_tile_fusing','OMEZarr')
        image_channel_zarrs = [x for x in os.listdir(stitched_zarrs) if os.path.isdir(os.path.join(stitched_zarrs,x)) and 'zarr' in x]
        image_channels = [x.split('.zarr')[0] for x in image_channel_zarrs]
        image_channels.pop(image_channels.index(alignment_channel))

        acquisition_path = f"{registration_data_asset}/acquisition.json"
        acquisition_json = read_json_as_dict(acquisition_path)
        acquisition_orientation = acquisition_json.get("axes")

        for this_channel in image_channels:
            # Load the channel
            this_ants_img = get_additional_channel_image_at_highest_level(
                os.path.join(registration_data_asset,'image_tile_fusing','OMEZarr',f'{this_channel}.zarr'),
                template,
                acquisition_orientation)
            ants.image_write(this_ants_img,os.path.join(image_histology_results,f'{this_channel}.nii.gz'))
            # 
            channel_in_ccf = ants.apply_transforms(ccf_25,this_ants_img,
                                      ['/data/spim_template_to_ccf/syn_1Warp.nii.gz',
                                       '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                       os.path.join(moved_image_folder,'ls_to_template_SyN_1Warp.nii.gz'),
                                       os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                                       ])
            ants.image_write(channel_in_ccf,os.path.join(histology_results,f'histology_{this_channel}.nrrd'))
                             
     # Tranform the CCF into image space
    ccf_in_image_space = ants.apply_transforms(zarr_read,
            ccf_25,
            [os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz'),
                '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
            whichtoinvert=[True,False,True,False],)
    ants.image_write(ccf_in_image_space,os.path.join(image_histology_results,f'ccf_in_{manifest_df.mouseid[0]}.nrrd'))

    ccf_labels_in_image_space = ants.apply_transforms(zarr_read,
                                        ccf_annotation_25,
                                        [os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                                            os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz'),
                                            '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                            '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
                                        whichtoinvert=[True,False,True,False],
                                        interpolator='genericLabel')
    ants.image_write(ccf_labels_in_image_space,os.path.join(image_histology_results,f'labels_in_{manifest_df.mouseid[0]}.nrrd'))
    
    # Prep file save local
    track_results = Path('/results/')/str(manifest_df.mouseid[0])/'track_data'
    os.makedirs(track_results,exist_ok= True)
    spim_results = os.path.join(track_results,'spim')
    os.makedirs(spim_results,exist_ok=True)
    template_results = os.path.join(track_results,'template')
    os.makedirs(template_results,exist_ok = True)
    ccf_results = os.path.join(track_results,'ccf')
    os.makedirs(ccf_results,exist_ok = True)
    bregma_results = os.path.join(track_results,'bregma_xyz')
    os.makedirs(bregma_results,exist_ok = True)                        
    
    processed_recordings = []

    for ii,row in manifest_df.iterrows():
        if row.annotation_format.lower()=='json':
            extension = 'json'
        else:
            raise ValueError('Currently only jsons from neuroglancer are supported!')

        # Find the sorted and origional data
        recording_id = row.sorted_recording.split('_sorted')[0]
        recording_folder = Path('/data/')/row.sorted_recording
        results_folder = Path('/results/')/str(row.mouseid)/recording_id

        pattern = f'*/{row.probe_file}.{extension}'
        annotation_file_path = next(Path('/data').glob(pattern), None)
        if annotation_file_path is None:
            print(f'Failed to find {pattern!r}')
            continue
        else:
            probe_data = read_neuroglancer_annotation_layers(annotation_file_path, layer_names = [row.probe_id])
            this_probe_data = pd.DataFrame({'x':probe_data[row.probe_id][:,0],
                                           'y':probe_data[row.probe_id][:,1],
                                           'z':probe_data[row.probe_id][:,2]})
            x = extrema[0]-this_probe_data.x.values*1e3+offset[0]
            y  = this_probe_data.y.values*1e3+offset[1]
            z = -this_probe_data.z.values*1e3+offset[2]

            this_probe = np.vstack([x,y,z]).T
            this_probe = order_annotation_pts(this_probe)
            create_slicer_fcsv(os.path.join(spim_results,f'{row.probe_id}.fcsv'),this_probe,direction = 'LPS')

            # Move probe into template space.
            this_probe_df = pd.DataFrame({'x':this_probe[:,0],'y':this_probe[:,1],'z':this_probe[:,2]})
            # Transform into template space
            this_probe_template = ants.apply_transforms_to_points(3,this_probe_df,[os.path.join(moved_image_folder,'ls_to_template_SyN_0GenericAffine.mat'),
                                                                                   os.path.join(moved_image_folder,'ls_to_template_SyN_1InverseWarp.nii.gz')],
                                                                               whichtoinvert=[True,False])
            create_slicer_fcsv(os.path.join(template_results,f'{row.probe_id}.fcsv'),this_probe_template.values,direction = 'LPS')

            # Move probe into ccf space
            this_probe_ccf = ants.apply_transforms_to_points(3,this_probe_template,['/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                                            '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz'],
                                           whichtoinvert=[True,False])
            create_slicer_fcsv(os.path.join(ccf_results,f'{row.probe_id}.fcsv'),this_probe_ccf.values,direction = 'LPS')

            # Transform into ibl x-y-z-picks space
            ccf_mlapdv = this_probe_ccf.values.copy()*1000
            ccf_mlapdv[:,0] = -ccf_mlapdv[:,0]
            ccf_mlapdv[:,1] = ccf_mlapdv[:,1]
            ccf_mlapdv[:,2] = -ccf_mlapdv[:,2]
            bregma_mlapdv = brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order='mlapdv')*1000000
            xyz_picks = {'xyz_picks':bregma_mlapdv.tolist()}

            if row.sorted_recording not in processed_recordings: # DEBUGGING HACK TO STOP EPHYS PROCESSING! FIX BEFORE RELEASE
                print(f'Have not yet processed: {row.sorted_recording}. Doing that now.') 
                os.makedirs(results_folder,exist_ok = True)
                try:
                    extract_spikes(recording_folder,results_folder)
                    extract_continuous(recording_folder,results_folder)
                except ValueError:
                    warnings.warn(f'Missing spike sorting for {row.sorted_recording}. Proceeding with histology only')
                    # Coppy the spike sorting error message to help future debugging.
                    shutil.copy(os.path.join(recording_folder,'output'),
                        os.path.join(results_folder,'output'))
                processed_recordings.append(row.sorted_recording)

            # Save this in two locations. First, save sorted by filename
            with open(os.path.join(bregma_results,f'{row.probe_id}.json'), "w") as f:
                # Serialize data to JSON format and write to file
                json.dump(xyz_picks, f)

            # Second, save the XYZ picks to the sorting folder for the gui.
            # This step will be skipped if there was a problem with the ephys pipeline.
            if os.path.isdir(os.path.join(results_folder,str(row.probe_name))):
                with open(os.path.join(results_folder,str(row.probe_name),'xyz_picks.json'),"w") as f:
                    json.dump(xyz_picks, f)
                             
