import os
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.tables import CSVTable

class DFTable(CSVTable):
    CSVTable.option_spec['df-arg'] = directives.unchanged
    df = None

    def __init__(self, name, arguments, options, content, lineno,
                 content_offset, block_text, state, state_machine):

        super().__init__(name, arguments, options, content, lineno,
                 content_offset, block_text, state, state_machine)

    def get_csv_data(self):
        return self.df.to_csv(index=False).splitlines(), None

    def run(self):
        source_file_name = self.state_machine.document.attributes["source"]
        dirname = os.path.abspath(os.path.dirname(source_file_name))
        os.chdir(dirname)

        code = "\n".join(map(str, self.content))
        ns = {}

        try:
            exec("\n".join( ["import numpy as np", "import pandas as pd", ]), ns)

            variable_name = "df"
            if self.options.get("df-var"):
                variable_name = self.options.get("df-var")

            exec(code, ns)
            self.df = ns[variable_name]
            
        except Exception as e:
            raise self.error(str(e))

        return super().run()

    
def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive("df-table", DFTable)

    metadata = {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": 0.1,
    }
    return metadata