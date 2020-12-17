from docutils import nodes
from docutils.parsers.rst import Directive, directives

import os, sys

def remove_symlink_handler(app, exception):
    dst = './src'
    
    if os.path.exists(dst):
        if os.path.isdir(dst):
            if os.path.islink(dst):
                 os.unlink(dst)
            else:
                shutil.rmtree(dst)
        else:
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                os.remove(dst)

def setup(app):
    app.connect('build-finished', remove_symlink_handler)
    src = '../src'
    dst = './src'

    # This creates a symbolic link on python in tmp directory

    if os.path.exists(dst):
        if os.path.isdir(dst):
            if os.path.islink(dst):
                 os.unlink(dst)
            else:
                shutil.rmtree(dst)
        else:
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                os.remove(dst)

    os.symlink(src, dst)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }