import os, glob, sys, subprocess

def prepare_code(repos, path):

    # --- Infer path 
    if path is None:

        # --- Check present directory parent(s)
        if os.getcwd().find('/dl_utils/') > -1:
            path = os.getcwd().split('/dl_utils')[0]

        # --- Check other user home directories
        elif len(glob.glob('/home/*/python/dl_utils')) > 0:
            path = sorted(glob.glob('/home/*/python/dl_utils'))[0]
            path = path.split('/dl_utils')[0]

        # --- Default: use home directory, python subdirectory
        else:
            path = '%s/python' % os.environ['HOME']
    
    for repo in repos:

        dst = '{}/dl_{}'.format(path, repo)

        # --- Pull rep
        if not os.path.exists(dst):
            args = ['git', 'clone', 'https://github.com/peterchang77/dl_{}'.format(repo), dst]
            subprocess.run(args)
            
        # --- Update repo
        else:
            args = ['git', '-C', dst, 'pull', 'origin', 'master']
            subprocess.run(args)
            
        # --- Add to sys.path
        if dst not in sys.path:
            sys.path.append(dst)

        # --- Add env variable
        os.environ['DL_{}_ROOT'.format(repo.upper())] = dst

def prepare_env(repos=['utils', 'train'], path=None, CUDA_VISIBLE_DEVICES=0):
    """
    Method to prepare dl_* environment

      (1) Update (or download) code repositories ==> set sys.path
      (2) Set CUDA devices

    :params

      (list) repos : list of repos to prepare; by default, dl_utils and dl_train
      (str)  path  : full path to parent dir of repositories; by default, $HOME/python

    """
    # --- Prepare Git repository code
    prepare_code(repos, path)

    # --- Set visible GPU if more than one is available on machine
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
