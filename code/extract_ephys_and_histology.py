import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os


import ants
import iblatlas.atlas as atlas
import json

from aind_morphology_utils.utils import read_swc

from extract_spikes import extract_spikes
from extract_continuous import extract_continuous

import argparse

def import_swc_probe_data(filename):
    S = read_swc(filename)
    return pd.DataFrame(S.compartment_list)

def create_slicer_fcsv(filename,pts_mat,direction = 'LPS',pt_orientation = [0,0,0,1],pt_visibility = 1,pt_selected = 0, pt_locked = 1):
    """
    Save fCSV file that is slicer readable.
    """
    # Create output file
    OutObj = open(filename,"w+")
    
    header0 = '# Markups fiducial file version = 4.11\n'
    header1 = '# CoordinateSystem = '+ direction+'\n'
    header2 = '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    
    OutObj.writelines([header0,header1,header2])
    
    outlines = []
    for ii in range(pts_mat.shape[0]):
        outlines.append(
            str(ii+1) +','+ 
            str(pts_mat[ii,0])+','+ 
            str(pts_mat[ii,1])+','+ 
            str(pts_mat[ii,2])+
            f',{pt_orientation[0]},{pt_orientation[1]},{pt_orientation[2]},{pt_orientation[3]},'+
            f'{pt_visibility},{pt_selected},{pt_locked},'+ 
            str(ii)+',,vtkMRMLScalarVolumeNode1\n')
    
    OutObj.writelines(outlines)
    OutObj.close()
    
def probe_df_to_fcsv(probe_data,extrema,results_folder,offset=(0,0,0)):
    unq = np.unique(probe_data.tree_id)
    probes = {}
    for ii,uu in enumerate(unq):
        this_probe_data = probe_data[probe_data.tree_id==uu]
        x = extrema[0]-(this_probe_data.x/1000).values+offset[0]
        y = (this_probe_data.y/1000).values+offset[1]
        z = -(this_probe_data.z/1000).values+offset[2]    
        probes[str(uu)] = np.vstack([x,y,z]).T
        create_slicer_fcsv(os.path.join(results_folder,f'test{uu}.fcsv'),probes[str(uu)],direction = 'LPS')
        
    return probes


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-sorted',
                        dest = 'sorting_folder',
                        default = 'ecephys_713506_2024-02-13_13-21-59_sorted_2024-02-14_08-03-38',
                        help = 'Sorted Folder to use as baseline')
    
    parser.add_argument('-probes',
                        dest = 'annotation_file',
                        default = '713506_annotations',
                        help = 'Directory containing probe annotations')
                        
    parser.add_argument('-manifest',
                        dest = 'annoation_manifest',
                        default = '713506_test.csv',
                        help = 'Probe Annotations')
                        
    parser.add_argument('-registration',
                    dest = 'registration_data',
                    default = '713506_to_ccf_Ex_639_Em_667_all_channel',
                    help = 'Directory containing output of registration')
                        
    args = parser.parse_args()
                        
    # If no values are passed, use default settings.
        
    if ('/data/' in args.sorting_folder):
        sorting_folder = args.sorting_folder.split('/data/')[-1]
    else:
        sorting_folder = args.sorting_folder
            
    if not ('/data/' in args.annotation_file):
        annotation_file_path = os.path.join('/data/',args.annotation_file)
    else:
        annotation_file_path = args.annotation_file
                        
    if not ('/data/' in args.annoation_manifest):
        annoation_manifest_path = os.path.join('/data/',args.annoation_manifest)
    else:
        annoation_manifest_path = args.annoation_manifest
        
    if not ('/data/' in args.registration_data):
        registration_data_asset = os.path.join('/data/',args.registration_data)
    else:
        registration_data_asset = args.registration_data
        
    # Read the annotation-ephys pairings
    df = pd.read_csv(annoation_manifest_path)

    # Load the template and the ccf
    template = ants.image_read('/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz')
    ccf_25 = ants.image_read('/data/allen_mouse_ccf/average_template/average_template_25.nii.gz')
    ccf_annotation_25 = ants.image_read('/data/allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz')
    brain_atlas = atlas.AllenAtlas(25,hist_path='/scratch/')
    
    # Get volume information to interpret probe tracks
    zarr_read = ants.image_read(os.path.join(registration_data_asset,'registration','prep_n4bias.nii.gz'))
    extrema = np.array(zarr_read.shape)*np.array(zarr_read.spacing)
    offset = zarr_read.origin
    
    # Get CCF space histology for this mouse
    histology_results = os.path.join('/results',str(df.mouseid[0]),'ccf_space_histology')
    os.makedirs(histology_results,exist_ok = True)

    outimg = ants.image_read(os.path.join(registration_data_asset,'registration','moved_ls_to_ccf.nii.gz'))
    ants.image_write(outimg,os.path.join(histology_results,f'histology_registration.nrrd'))

    other_files = [x for x in os.listdir(os.path.join(registration_data_asset,'registration')) if 'moved_ls_to_template_' in x and '.nii.gz' in x]
    for fl in other_files:
        chname = fl.split('moved_ls_to_template_')[-1].split('.nii.gz')[0]
        image_in_template = ants.image_read(os.path.join(registration_data_asset,'registration',fl))
        outimg = ants.apply_transforms(ccf_25,image_in_template,['/data/spim_template_to_ccf/syn_1Warp.nii.gz','/data/spim_template_to_ccf/syn_0GenericAffine.mat'])
        ants.image_write(outimg,os.path.join(histology_results,f'histology_{chname}.nrrd'))
        
    # Prep file save local
    track_results = Path('/results/')/str(df.mouseid[0])/'track_data'
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

    for ii,row in df.iterrows():
        if row.annotation_format.lower()=='swc':
            extension = 'swc'
        else:
            raise ValueError('Currently only swc annotations from horta are supported!')

        recording_id = row.sorted_recording.split('_sorted')[0]
        recording_folder = Path('/data/')/row.sorted_recording
        results_folder = Path('/results/')/str(row.mouseid)/recording_id

        if not os.path.exists(Path(annotation_file_path)/f'{row.probe_file}.{extension}'):
            missing = Path(annotation_file_path)/f'{row.probe_file}.{extension}'
            print(f'Failed to find {missing}')
            continue
        else:
            print(row.probe_file)
            probe_data = import_swc_probe_data(Path(annotation_file_path)/f'{row.probe_file}.{extension}')

        # do the preprocessing for all channels in the given recording
        # Any errors here are likely due files not being found. 
        # Check that the correct data are attached to the capsual!
        if row.sorted_recording not in processed_recordings:
            print(f'Have not yet processed: {row.sorted_recording}. Doing that now.') 
            os.makedirs(results_folder,exist_ok = True)
            extract_spikes(recording_folder,results_folder)
            extract_continuous(recording_folder,results_folder)
            processed_recordings.append(row.sorted_recording)

        # Get relevent subset of data. Usefule if more than one probe in file...but we may cut this later.
        this_probe_data = probe_data[probe_data.tree_id==row.probe_id]
        if np.any(probe_data.tree_id.values>0):
            probe_name = row.probe_file+'_'+row.probe_id
        else:
            probe_name = row.probe_file

        # Get probe in spim space.
        # This math handles different readout conventions.
        x = extrema[0]-(this_probe_data.x/1000).values+offset[0]
        y = (this_probe_data.y/1000).values+offset[1]
        z = -(this_probe_data.z/1000).values+offset[2]    
        this_probe = np.vstack([x,y,z]).T
        create_slicer_fcsv(os.path.join(spim_results,f'{probe_name}.fcsv'),this_probe,direction = 'LPS')

        # Move probe into template space.
        this_probe_df = pd.DataFrame({'x':this_probe[:,0],'y':this_probe[:,1],'z':this_probe[:,2]})
        # Transform into template space
        this_probe_template = ants.apply_transforms_to_points(3,this_probe_df,[os.path.join(registration_data_asset,'registration','ls_to_template_SyN_0GenericAffine.mat'),
                                                                               os.path.join(registration_data_asset,'registration','ls_to_template_SyN_1InverseWarp.nii.gz')],
                                                                           whichtoinvert=[True,False])
        create_slicer_fcsv(os.path.join(template_results,f'{probe_name}.fcsv'),this_probe_template.values,direction = 'LPS')

        # Move probe into ccf space
        this_probe_ccf = ants.apply_transforms_to_points(3,this_probe_template,['/data/spim_template_to_ccf/syn_0GenericAffine.mat',
                                                        '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz'],
                                       whichtoinvert=[True,False])
        create_slicer_fcsv(os.path.join(ccf_results,f'{probe_name}.fcsv'),this_probe_ccf.values,direction = 'LPS')

        # Transform into ibl x-y-z-picks space
        ccf_mlapdv = this_probe_ccf.values.copy()*1000
        ccf_mlapdv[:,0] = -ccf_mlapdv[:,0]
        ccf_mlapdv[:,1] = ccf_mlapdv[:,1]
        ccf_mlapdv[:,2] = -ccf_mlapdv[:,2]
        bregma_mlapdv = brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order='mlapdv')*1000000
        xyz_picks = {'xyz_picks':bregma_mlapdv.tolist()}

        # Save this in two locations. First, save sorted by filename
        with open(os.path.join(bregma_results,f'{probe_name}.json'), "w") as f:
            # Serialize data to JSON format and write to file
            json.dump(xyz_picks, f)

        with open(os.path.join(results_folder,str(row.probe_name),'xyz_picks.json'),"w") as f:
            json.dump(xyz_picks, f)