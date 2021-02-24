import os, tarfile

def archive_handler(app):
    old_cwd = os.getcwd()
    src_dir = 'src'
    libtorch_dir = 'libtorch_demo'
    archive_name = libtorch_dir + '.tar.gz'

    os.chdir(src_dir)

    try:
        os.remove(archive_name)
    except OSError:
        pass

    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(libtorch_dir)

    os.chdir(old_cwd)

def setup(app):
    app.connect('builder-inited', archive_handler)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
