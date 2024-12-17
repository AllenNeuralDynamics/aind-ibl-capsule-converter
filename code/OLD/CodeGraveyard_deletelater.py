#     # Get CCF space histology for this mouse
#     histology_results = os.path.join('/results',str(df.mouseid[0]),'ccf_space_histology')
#     os.makedirs(histology_results,exist_ok = True)

#     outimg = ants.image_read(os.path.join(registration_data_asset,'registration','moved_ls_to_ccf.nii.gz'))
#     ants.image_write(outimg,os.path.join(histology_results,f'histology_registration.nrrd'))

#     other_files = [x for x in os.listdir(os.path.join(registration_data_asset,'registration')) if 'moved_ls_to_template_' in x and '.nii.gz' in x]
#     for fl in other_files:
#         chname = fl.split('moved_ls_to_template_')[-1].split('.nii.gz')[0]
#         image_in_template = ants.image_read(os.path.join(registration_data_asset,'registration',fl))
#         outimg = ants.apply_transforms(ccf_25,image_in_template,['/data/spim_template_to_ccf/syn_1Warp.nii.gz','/data/spim_template_to_ccf/syn_0GenericAffine.mat'])
#         ants.image_write(outimg,os.path.join(histology_results,f'histology_{chname}.nrrd'))
    

#     # Also get the image-space warping of the CCF.
#     image_histology_results = os.path.join('/results',str(df.mouseid[0]),'image_space_histology')
#     os.makedirs(image_histology_results,exist_ok = True)

#     ccf_in_image_space = ants.apply_transforms(zarr_read,
#                                             ccf_25,
#                                             [os.path.join(registration_data_asset,'registration','ls_to_template_SyN_0GenericAffine.mat'),
#                                                 os.path.join(registration_data_asset,'registration','ls_to_template_SyN_1InverseWarp.nii.gz'),
#                                                 '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
#                                                 '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
#                                             whichtoinvert=[True,False,True,False],)
#     ants.image_write(ccf_in_image_space,os.path.join(image_histology_results,f'ccf_in_{df.mouseid[0]}.nrrd'))

#     ccf_labels_in_image_space = ants.apply_transforms(zarr_read,
#                                             ccf_annotation_25,
#                                             [os.path.join(registration_data_asset,'registration','ls_to_template_SyN_0GenericAffine.mat'),
#                                                 os.path.join(registration_data_asset,'registration','ls_to_template_SyN_1InverseWarp.nii.gz'),
#                                                 '/data/spim_template_to_ccf/syn_0GenericAffine.mat',
#                                                 '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz',],
#                                             whichtoinvert=[True,False,True,False],
#                                             interpolator='genericLabel')
#     ants.image_write(ccf_labels_in_image_space,os.path.join(image_histology_results,f'labels_in_{df.mouseid[0]}.nrrd'))

#     # and copy the image space data into this folder
#     shutil.copy(os.path.join(registration_data_asset,'registration','prep_n4bias.nii.gz'),
#                 os.path.join(image_histology_results,'histology_registration.nii.gz'))
#     for fl in other_files:
#         chname = fl.split('moved_ls_to_template_')[-1].split('.nii.gz')[0]
#         shutil.copy(os.path.join(registration_data_asset,'registration',fl),os.path.join(image_histology_results,'histology_{chname}.nii.gz'))


#     # Prep file save local
#     track_results = Path('/results/')/str(df.mouseid[0])/'track_data'
#     os.makedirs(track_results,exist_ok= True)
#     spim_results = os.path.join(track_results,'spim')
#     os.makedirs(spim_results,exist_ok=True)
#     template_results = os.path.join(track_results,'template')
#     os.makedirs(template_results,exist_ok = True)
#     ccf_results = os.path.join(track_results,'ccf')
#     os.makedirs(ccf_results,exist_ok = True)
#     bregma_results = os.path.join(track_results,'bregma_xyz')
#     os.makedirs(bregma_results,exist_ok = True)

#     processed_recordings = []

#     for ii,row in df.iterrows():
#         if row.annotation_format.lower()=='swc':
#             extension = 'swc'
#         elif row.annotaiton_format.lower()=='json':
#             extension = 'json'
#         else:
#             raise ValueError('Currently only swc annotations from horta OR jsons from neuroglancer are supported!')

#         recording_id = row.sorted_recording.split('_sorted')[0]
#         recording_folder = Path('/data/')/row.sorted_recording
#         results_folder = Path('/results/')/str(row.mouseid)/recording_id
        
        
#         if not os.path.exists(Path(annotation_file_path)/f'{row.probe_file}.{extension}'):
#             missing = Path(annotation_file_path)/f'{row.probe_file}.{extension}'
#             print(f'Failed to find {missing}')
#             continue
#         else:
#             if extension == 'swc'
#                 print(row.probe_file)
#                 probe_data = import_swc_probe_data(Path(annotation_file_path)/f'{row.probe_file}.{extension}')
#             else 
                

#         # do the preprocessing for all channels in the given recording
#         # Any errors here are likely due files not being found. 
#         # Check that the correct data are attached to the capsual!
#         if row.sorted_recording not in processed_recordings:
#             print(f'Have not yet processed: {row.sorted_recording}. Doing that now.') 
#             os.makedirs(results_folder,exist_ok = True)
#             extract_spikes(recording_folder,results_folder)
#             extract_continuous(recording_folder,results_folder)
#             processed_recordings.append(row.sorted_recording)

#         # Get relevent subset of data. Usefule if more than one probe in file...but we may cut this later.
#         this_probe_data = probe_data[probe_data.tree_id==row.probe_id]
#         if np.any(probe_data.tree_id.values>0):
#             probe_name = row.probe_file+'_'+row.probe_id
#         else:
#             probe_name = row.probe_file

#         # Get probe in spim space.
#         # This math handles different readout conventions.
#         x = extrema[0]-(this_probe_data.x/1000).values+offset[0]
#         y = (this_probe_data.y/1000).values+offset[1]
#         z = -(this_probe_data.z/1000).values+offset[2]    
#         this_probe = np.vstack([x,y,z]).T
#         create_slicer_fcsv(os.path.join(spim_results,f'{probe_name}.fcsv'),this_probe,direction = 'LPS')

#         # Move probe into template space.
#         this_probe_df = pd.DataFrame({'x':this_probe[:,0],'y':this_probe[:,1],'z':this_probe[:,2]})
#         # Transform into template space
#         this_probe_template = ants.apply_transforms_to_points(3,this_probe_df,[os.path.join(registration_data_asset,'registration','ls_to_template_SyN_0GenericAffine.mat'),
#                                                                                os.path.join(registration_data_asset,'registration','ls_to_template_SyN_1InverseWarp.nii.gz')],
#                                                                            whichtoinvert=[True,False])
#         create_slicer_fcsv(os.path.join(template_results,f'{probe_name}.fcsv'),this_probe_template.values,direction = 'LPS')

#         # Move probe into ccf space
#         this_probe_ccf = ants.apply_transforms_to_points(3,this_probe_template,['/data/spim_template_to_ccf/syn_0GenericAffine.mat',
#                                                         '/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz'],
#                                        whichtoinvert=[True,False])
#         create_slicer_fcsv(os.path.join(ccf_results,f'{probe_name}.fcsv'),this_probe_ccf.values,direction = 'LPS')

#         # Transform into ibl x-y-z-picks space
#         ccf_mlapdv = this_probe_ccf.values.copy()*1000
#         ccf_mlapdv[:,0] = -ccf_mlapdv[:,0]
#         ccf_mlapdv[:,1] = ccf_mlapdv[:,1]
#         ccf_mlapdv[:,2] = -ccf_mlapdv[:,2]
#         bregma_mlapdv = brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order='mlapdv')*1000000
#         xyz_picks = {'xyz_picks':bregma_mlapdv.tolist()}

#         # Save this in two locations. First, save sorted by filename
#         with open(os.path.join(bregma_results,f'{probe_name}.json'), "w") as f:
#             # Serialize data to JSON format and write to file
#             json.dump(xyz_picks, f)

#         with open(os.path.join(results_folder,str(row.probe_name),'xyz_picks.json'),"w") as f:
#             json.dump(xyz_picks, f)