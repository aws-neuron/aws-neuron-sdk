import os
import sys

from sphinx.ext.autodoc import ModuleDocumenter, FunctionDocumenter


class LocalModuleDocumenter(ModuleDocumenter):
    """
    Provides identical functionality to "automodule", but allows the module
    function names to be overridden with the "module-name" option.

    This also allows local python files to be documented as if they were
    imported from an actual package by temporarily adding the directory of the
    RST file to the python path.
    """
    option_spec = dict(ModuleDocumenter.option_spec)
    option_spec['module-name'] = lambda x = None: x

    def import_object(self, *args):
        """Find modules local to the RST document directory"""
        local = os.path.join(self.env.app.srcdir, os.path.dirname(self.env.docname))
        sys.path.append(local)
        result = super().import_object(*args)
        sys.path.remove(local)
        return result

    def get_module_members(self):
        """Add module name override to local files"""
        members = super().get_module_members()
        name = self.options.module_name
        if name is not None:
            for member in members.values():
                if callable(member.object):
                    setattr(member.object, 'module_name_override', name)
        return members


class LocalFunctionDocumenter(FunctionDocumenter):
    def format_name(self) -> str:
        """Apply module name override to local functions"""
        # Use overridden module path if it is provided
        if hasattr(self.object, 'module_name_override'):
            self.objpath = self.object.module_name_override.split('.') + [self.objpath[-1]]
        return super().format_name()


def setup(app):
    app.add_autodocumenter(LocalFunctionDocumenter)
    app.add_autodocumenter(LocalModuleDocumenter)
