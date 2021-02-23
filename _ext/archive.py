import os, tarfile

def archive_handler(app, exception):
    old_cwd = os.getcwd()
    libtorch = 'neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/libtorch'
    src_name = 'libtorch_demo'
    archive_name = src_name + '.tar.gz'

    os.chdir(libtorch)

    try:
        os.remove(archive_name)
    except OSError:
        pass

    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(src_name)

    os.chdir(old_cwd)

def setup(app):
    app.connect('build-finished', archive_handler)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
