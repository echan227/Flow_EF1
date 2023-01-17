import os

root_dir = '/data_GSTT/Diak_data/AMC_XPLORE_analysis/new_data_june2022/AMC/nifti/'
subjects = sorted([fname for fname in os.listdir(root_dir)])
# print(subjects)

seqs = ['SAX', 'LAX']
for seq in seqs:
    out_dir = '/data_GSTT/Diak_data/AMC_XPLORE_analysis/new_data_june2022/AMC/{0}_gifs/'.format(seq)
    for d, subject_id in enumerate(subjects):
        data_dir  = os.path.join(root_dir, subject_id, 'results')
        if os.path.exists(os.path.join(data_dir, '{0}_{1}.gif'.format(subject_id,seq))):
            print('{0}: {1}'.format(d, subject_id))
            save_dir = os.path.join(out_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            os.system('cp {} {}'.format(os.path.join(data_dir, '{0}_{1}.gif'.format(subject_id,seq)), os.path.join(save_dir, '{0}_{1}.gif'.format(subject_id,seq))))
            print('cp {} {}'.format(os.path.join(data_dir, '{0}_{1}.gif'.format(subject_id,seq)), os.path.join(save_dir, '{0}_{1}.gif'.format(subject_id,seq))))
            #os.system('cp {} {}'.format(os.path.join(data_dir, 'la_4Ch.nii.gz'), os.path.join(save_dir, 'la_4Ch.nii.gz')))
            #print('cp {} {}'.format(os.path.join(data_dir, 'la_4Ch.nii.gz'), os.path.join(save_dir, 'la_4Ch.nii.gz')))
