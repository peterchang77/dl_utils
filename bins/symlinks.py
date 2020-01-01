import os, glob, argparse

def find_files(root):
    """
    Method to find all files except:

      *.pyc
      *.swp
      *.zip

    """
    IGNORE = ['pyc', 'swp', 'zip']

    files = glob.glob('%s/**/*' % root, recursive=True)
    files = [f for f in files if f.split('.')[-1] not in IGNORE]

    return files

def update_symlinks(src, dst, remove):
    """
    Method to create symlinks for *.py from [src] to [dst]

    """
    IGNORE = []
    src = os.path.abspath(src)
    dst = os.path.abspath(dst or '')

    # =======================================================
    # (1) | REMOVE ONLY
    # =======================================================

    if remove:

        files = find_files(src) 
        remove_symlinks(files)

        return

    # =======================================================
    # (2) | REMOVE + UPDATE 
    # =======================================================

    src_files = find_files(src) 
    dst_files = [s.replace(src, dst) for s in src_files]

    # --- Remove existing symlinks
    remove_symlinks(dst_files)

    for src_file, dst_file in zip(src_files, dst_files):

        if os.path.basename(dst_file) in IGNORE or os.path.exists(dst_file):
            print('Skipping symlink: %s' % dst_file)

        else:
            print('Creating symlink: %s' % dst_file)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            os.symlink(src=src_file, dst=dst_file)

def remove_symlinks(files):
    """
    Method to remove symlinks from list of files 

    """
    for f in files:
        if os.path.islink(f):
            os.remove(f)

if __name__ == '__main__':

    description = 'Create symlinks for *.py from [src] to [dst] while preserving directory structure.'
    usage = 'symlinks [-h] [--remove] src [, dst]'

    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('-r', '--remove', action='store_true', default=False)
    parser.add_argument('src', metavar='src', type=str, help='source directory (./lite/, etc...)')
    parser.add_argument('dst', metavar='dst', type=str, nargs='?', default=None, help='target directory (./full/, etc...)')
    args = parser.parse_args()

    update_symlinks(args.src, args.dst, args.remove)

    pass


