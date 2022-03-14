# This file creates a downloadable archive from each directory listed in src_dirs.
# You can modify or add additional archive_handler functions here to create additional archives.

import os, tarfile

def archive_handler(app):
    old_cwd = os.getcwd()
    src_dirs = ['src/examples/pytorch', 'src']
    target_dirs = ['libtorch_demo', 'neuronperf']
    archive_names = [name + '.tar.gz' for name in target_dirs]

    for src_dir, target_dir, archive_name in zip(src_dirs, target_dirs, archive_names):
        os.chdir(src_dir)

        try:
            os.remove(archive_name)
        except OSError:
            pass

        with tarfile.open(archive_name, 'w:gz') as tar:
            tar.add(target_dir)

        os.chdir(old_cwd)

def setup(app):
    app.connect('builder-inited', archive_handler)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
