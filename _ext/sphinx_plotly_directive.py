"""
CODE FROM: https://github.com/harupy/sphinx-plotly-directive
LICENSE: MIT

Based on: https://matplotlib.org/3.1.3/devel/plot_directive.html

A directive for including a Plotly figure in a Sphinx document
================================================================

By default, in HTML output, `plot` will include a .png file with a link to a
high-res .png and .pdf.  In LaTeX output, it will include a .pdf.

The source code for the plot may be included in one of three ways:

1. **A path to a source file** as the argument to the directive::

     .. plot:: path/to/plot.py

   When a path to a source file is given, the content of the
   directive may optionally contain a caption for the plot::

     .. plot:: path/to/plot.py

        The plot's caption.

   Additionally, one may specify the name of a function to call (with
   no arguments) immediately after importing the module::

     .. plot:: path/to/plot.py plot_function1

2. Included as **inline content** to the directive::

     .. plotly::

        import plotly.express as px
        px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

3. Using **doctest** syntax::

     .. plotly::

        A plotting example:
        >>> import plotly.express as px
        >>> px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

4. Using the `fig-vars` option. In the example below, `fig1` and `fig2` will be
   rendered::

     .. plotly::
        :fig-vars: fig1, fig2

        import plotly.express as px
        fig1 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig2 = px.scatter(x=[4, 3, 2, 1, 0], y=[0, 1, 4, 9, 16])

Options
-------

The ``plotly`` directive supports the following options:

    format : {'python', 'doctest'}
        The format of the input.

    include-source : bool
        Whether to display the source code. The default can be changed
        using the `plot_include_source` variable in :file:`conf.py`.

    encoding : str
        If this source file is in a non-UTF8 or non-ASCII encoding, the
        encoding must be specified using the ``:encoding:`` option.  The
        encoding will not be inferred using the ``-*- coding -*-`` metacomment.

    context : bool or str
        If provided, the code will be run in the context of all previous plot
        directives for which the ``:context:`` option was specified.  This only
        applies to inline code plot directives, not those run from files. If
        the ``:context: reset`` option is specified, the context is reset
        for this and future plots, and previous figures are closed prior to
        running the code. ``:context: close-figs`` keeps the context but closes
        previous figures before running the code.

    nofigs : bool
        If specified, the code block will be run, but no figures will be
        inserted.  This is usually useful with the ``:context:`` option.

    caption : str
        If specified, the option's argument will be used as a caption for the
        figure. This overwrites the caption given in the content, when the plot
        is generated from a file.

    iframe-width
        The width of the iframe in which a plotly figure is rendered. The default can be changed
        using the `plotly_iframe_width` variable in :file:`conf.py`.

    iframe-height
        The height of the iframe in which a plotly figure is rendered. The default can be changed
        using the `plotly_iframe_height` variable in :file:`conf.py`.

Additionally, this directive supports all of the options of the `image`
directive, except for *target* (since plot will add its own target).  These
include *alt*, *height*, *width*, *scale*, *align* and *class*.

Configuration options
---------------------

The plot directive has the following configuration options:

    plotly_include_source
        Default value for the include-source option

    plotly_html_show_source_link
        Whether to show a link to the source in HTML.

    plotly_pre_code
        Code that should be executed before each plot. If not specified or None
        it will default to a string containing::

            import numpy as np
            import plotly
            import plotly.graph_objects as go
            import plotly.express as px

    plotly_basedir
        Base directory, to which ``plot::`` file names are relative
        to.  (If None or empty, file names are relative to the
        directory where the file containing the directive is.)

    plotly_formats
        File formats to generate. List of tuples or strings::

            [(suffix, dpi), suffix, ...]

        that determine the file format and the DPI. For entries whose
        DPI was omitted, sensible defaults are chosen. When passing from
        the command line through sphinx_build the list should be passed as
        suffix:dpi,suffix:dpi, ...

    plotly_html_show_formats
        Whether to show links to the files in HTML.

    plotly_working_directory
        By default, the working directory will be changed to the directory of
        the example, so the code can get at its data files, if any.  Also its
        path will be added to `sys.path` so it can import any helper modules
        sitting beside it.  This configuration option can be used to specify
        a central directory (also added to `sys.path`) where data files and
        helper modules for all code are located.

    plotly_iframe_width
        The width of the iframe in which a plotly figure is rendered. The default is "100%".

    plotly_iframe_height
        The height of the iframe in which a plotly figure is rendered. The default is "500px".

    plotly_template
        Provide a customized template for preparing restructured text.
"""

import copy
import itertools
import os
import re
import shutil
import textwrap
import traceback
from os.path import relpath
from pathlib import Path

import jinja2  # Sphinx dependency.
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.images import Image

import re
import textwrap

import plotly


INDENT_SPACES = " " * 3


def save_plotly_figure(fig, path):
    r"""
    Save a Plotly figure.
    Parameters
    ----------
    fig : plotly figure
        A plotly figure to save.
    path : str
        A file path.
    Returns
    -------
    None
    Examples
    --------
    >>> import plotly.express as px
    >>> import tempfile
    >>> fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    >>> path = tempfile.NamedTemporaryFile(suffix=".html").name
    >>> save_plotly_figure(fig, path)
    """
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)
    with open(path, "w") as f:
        f.write(fig_html)


def assign_last_line_into_variable(code, variable_name):
    r"""
    Save a Plotly figure.
    Parameters
    ----------
    code : str
        A string representing code.
    name : str
        A variable name.
    Returns
    -------
    str
        Mew code.
    Examples
    --------
    >>> code = "a = 1\nfunc(a)"
    >>> new_code = assign_last_line_into_variable(code, "b")
    >>> print(new_code)
    a = 1
    b = func(a)
    """
    lines = code.split("\n")
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip() != "":
            lines[idx] = "{} = ".format(variable_name) + lines[idx]
            break
    return "\n".join(lines)


def create_directive_block(name, arguments, options, content):
    r"""
    Create a directive block.
    Parameters
    ----------
    name : str
        A directive name.
    arguments : list of str
        Arguments of the directive.
    option : dict
        Option of the directive.
    content : list of str
        Content of the directive.
    Returns
    -------
    str
        A directive block.
    Examples
    --------
    >>> block = create_directive_block(
    ...     "plotly",
    ...     ["f1", "f2"],
    ...     {"a": 0, "b": 1},
    ...     ["l1", "l2"],
    ... )
    >>> print(block)
    .. plotly:: f1 f2
       :a: 0
       :b: 1
    <BLANKLINE>
       l1
       l2
    """
    header = ".. {}:: ".format(name) + " ".join(arguments)
    code = "\n".join(map(str, content))

    lines = [header]

    if len(options.items()) > 0:

        def process_value(v):
            if isinstance(v, list):
                return ", ".join(v)
            return v

        options_block = "\n".join(":{}: {}".format(k, process_value(v)) for k, v in options.items())
        lines.append(textwrap.indent(options_block, INDENT_SPACES))

    lines.append("")
    lines.append(textwrap.indent(code, INDENT_SPACES))

    return "\n".join(lines)


def create_code_block(code, language=None):
    return "\n".join(
        [
            ".. code-block::{}".format(" " + language if language else ""),
            "",
            textwrap.indent(code.strip(), INDENT_SPACES),
            "",
        ]
    )


def strip_last_line(code):
    r"""
    Strips the last line of the give code block
    Parameters
    ----------
    code : str
        Code to strip
    Returns
    -------
    str:
        Stripped code
    Examples
    --------
    >>> strip_last_line("a")
    ''
    >>> strip_last_line("a\nb")
    'a'
    >>> strip_last_line("a\nb\nc")
    'a\nb'
    """
    return "\n".join(code.strip().split("\n")[:-1])


def ends_with_show(code):
    r"""
    Returns True if the last line of the given code block ends with `show()`
    Parameters
    ----------
    code : str
        Code that may contain a line that looks like `fig.show()`
    Returns
    -------
    str:
        Variable name of the object that calls `show()`
    Examples
    --------
    >>> ends_with_show("fig.show()")  # simple
    True
    >>> ends_with_show("fig.show(1, a=2)")  # show with arguments
    True
    >>> ends_with_show("fig = dummy\nfig.show()\n")  # multiline
    True
    >>> ends_with_show("foo")  # doesn't contains `show`
    False
    """
    # TODO: Use a more strict regular expression
    pattern = r"^(.+)\.show\(.*\)$"
    match = re.search(pattern, code.strip().split("\n")[-1], flags=re.DOTALL)
    return bool(match)


# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ("no", "0", "false"):
        return False
    elif arg.strip().lower() in ("yes", "1", "true"):
        return True
    else:
        raise ValueError('"%s" unknown boolean' % arg)


def _option_context(arg):
    if arg in [None, "reset", "close-figs"]:
        return arg
    raise ValueError("Argument should be None or 'reset' or 'close-figs'")


def _option_format(arg):
    return directives.choice(arg, ("python", "doctest"))


def _option_fig_vars(arg):
    return [x.strip() for x in arg.split(",")]


def mark_plot_labels(app, document):
    """
    To make plots referenceable, we need to move the reference from the
    "htmlonly" (or "latexonly") node to the actual figure node itself.
    """
    for name, explicit in document.nametypes.items():
        if not explicit:
            continue
        labelid = document.nameids[name]
        if labelid is None:
            continue
        node = document.ids[labelid]
        if node.tagname in ("html_only", "latex_only"):
            for n in node:
                if n.tagname == "figure":
                    sectname = name
                    for c in n:
                        if c.tagname == "caption":
                            sectname = c.astext()
                            break

                    node["ids"].remove(labelid)
                    node["names"].remove(name)
                    n["ids"].append(labelid)
                    n["names"].append(name)
                    document.settings.env.labels[name] = (
                        document.settings.env.docname,
                        labelid,
                        sectname,
                    )
                    break


class PlotlyDirective(Directive):
    """The ``.. plotly::`` directive, as documented in the module's docstring."""

    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        "alt": directives.unchanged,
        "height": directives.length_or_unitless,
        "width": directives.length_or_percentage_or_unitless,
        "scale": directives.nonnegative_int,
        "align": Image.align,
        "class": directives.class_option,
        "include-source": _option_boolean,
        "format": _option_format,
        "context": _option_context,
        "nofigs": directives.flag,
        "encoding": directives.encoding,
        "caption": directives.unchanged,
        "fig-vars": _option_fig_vars,
        "iframe-width": directives.unchanged,
        "iframe-height": directives.unchanged,
    }

    def run(self):
        """Run the plot directive."""
        try:
            return run(
                self.arguments,
                self.content,
                self.options,
                self.state_machine,
                self.state,
                self.lineno,
            )
        except Exception as e:
            raise self.error(str(e))


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive("plotly", PlotlyDirective)
    app.add_config_value("plotly_pre_code", None, True)
    app.add_config_value("plotly_include_source", False, True)
    app.add_config_value("plotly_html_show_source_link", True, True)
    app.add_config_value("plotly_formats", ["html"], True)
    app.add_config_value("plotly_basedir", None, True)
    app.add_config_value("plotly_html_show_formats", True, True)
    app.add_config_value("plotly_working_directory", None, True)
    app.add_config_value("plotly_iframe_width", "100%", True)
    app.add_config_value("plotly_iframe_height", "500px", True)
    app.add_config_value("plotly_template", None, True)

    app.add_config_value("plotly_include_directive_source", None, False)

    app.connect("doctree-read", mark_plot_labels)

    metadata = {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": 0.1,
    }
    return metadata


# -----------------------------------------------------------------------------
# Doctest handling
# -----------------------------------------------------------------------------


def contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, "<string>", "exec")
        return False
    except SyntaxError:
        pass
    r = re.compile(r"^\s*>>>", re.M)
    m = r.search(text)
    return bool(m)


def unescape_doctest(text):
    """
    Extract code from a piece of text, which contains either Python code
    or doctests.
    """
    if not contains_doctest(text):
        return text

    code = ""
    for line in text.split("\n"):
        m = re.match(r"^\s*(>>>|\.\.\.) (.*)$", line)
        if m:
            code += m.group(2) + "\n"
        elif line.strip():
            code += "# " + line.strip() + "\n"
        else:
            code += "\n"
    return code


def split_code_at_show(text):
    """Split code at plt.show()."""
    parts = []
    is_doctest = contains_doctest(text)

    part = []
    for line in text.split("\n"):
        if (not is_doctest and line.strip() == "plt.show()") or (
            is_doctest and line.strip() == ">>> plt.show()"
        ):
            part.append(line)
            parts.append("\n".join(part))
            part = []
        else:
            part.append(line)
    if "\n".join(part).strip():
        parts.append("\n".join(part))
    return parts


# -----------------------------------------------------------------------------
# Template
# -----------------------------------------------------------------------------


TEMPLATE = """
{% if directive_source %}
Source:

{{ directive_source }}

Output:
{% endif %}
{{ source_code }}

.. only:: html

   {% if source_link or (html_show_formats and not multi_image) %}
   (
   {%- if source_link -%}
   `Source code <{{ source_link }}>`__
   {%- endif -%}
   {%- if html_show_formats and not multi_image -%}
     {%- for fig in figures -%}
       {%- for fmt in fig.formats -%}
         {%- if source_link or not loop.first -%}, {% endif -%}
         `{{ fmt }} <{{ dest_dir }}/{{ fig.basename }}.{{ fmt }}>`__
       {%- endfor -%}
     {%- endfor -%}
   {%- endif -%}
   )
   {% endif %}

   {% for fig in figures %}
   .. raw:: html
      {% for option in options -%}
      {{ option }}
      {% endfor %}

       <iframe src="{{ fig.basename }}.{{ default_fmt }}" width="{{ iframe_width }}"
        height="{{ iframe_height }}" frameborder="0"></iframe>

   {% if html_show_formats and multi_figure -%}
     (
     {%- for fmt in fig.formats -%}
     {%- if not loop.first -%}, {% endif -%}
     `{{ fmt }} <{{ dest_dir }}/{{ fig.basename }}.{{ fmt }}>`__
     {%- endfor -%}
     )
   {%- endif -%}

      {{ caption }}
   {% endfor %}

.. only:: not html

   {% for fig in figures %}
   .. raw:: html
      {% for option in options -%}
      {{ option }}
      {% endfor %}

       <iframe src="{{ fig.basename }}.{{ default_fmt }}" width="{{ iframe_width }}"
        height="{{ iframe_height }}" frameborder="0"></iframe>

      {{ caption }}
   {% endfor %}

"""

exception_template = """
.. only:: html

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# the context of the plot for all directives specified with the
# :context: option
plot_context = dict()


class FigureFile:
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, "%s.%s" % (self.basename, format))

    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]


def out_of_date(original, derived):
    """
    Return whether *derived* is out-of-date relative to *original*, both of
    which are full file paths.
    """
    return not os.path.exists(derived) or (
        os.path.exists(original) and os.stat(derived).st_mtime < os.stat(original).st_mtime
    )


class PlotError(RuntimeError):
    pass


def run_code(code, code_path, ns=None, function_name=None, fig_vars=None):
    """
    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """

    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.  Add its path to sys.path
    # so it can import any helper modules sitting beside it.
    pwd = os.getcwd()
    if setup.config.plotly_working_directory is not None:
        try:
            os.chdir(setup.config.plotly_working_directory)
        except OSError as err:
            raise OSError(
                str(err) + "\n`plot_working_directory` option in"
                "Sphinx configuration file must be a valid "
                "directory path"
            ) from err
        except TypeError as err:
            raise TypeError(
                str(err) + "\n`plot_working_directory` option in "
                "Sphinx configuration file must be a string or "
                "None"
            ) from err
    elif code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)

    try:
        code = unescape_doctest(code)
        if ns is None:
            ns = {}
        if not ns:
            if setup.config.plotly_pre_code is None:
                exec(
                    "\n".join(
                        [
                            "import numpy as np",
                            "import plotly",
                            "import plotly.graph_objects as go",
                            "import plotly.express as px",
                        ]
                    ),
                    ns,
                )
            else:
                exec(str(setup.config.plotly_pre_code), ns)
        if "__main__" in code:
            ns["__name__"] = "__main__"

        variable_name = "fig"

        if ends_with_show(code):
            exec(strip_last_line(code), ns)
            figs = [ns[fig_var] for fig_var in fig_vars] if fig_vars else [ns[variable_name]]
        elif function_name is not None:
            exec(code, ns)
            exec(assign_last_line_into_variable(function_name + "()", variable_name), ns)
            figs = [ns[variable_name]]
        elif fig_vars:
            exec(code, ns)
            figs = [ns[fig_var] for fig_var in fig_vars]
        else:
            exec(assign_last_line_into_variable(code, variable_name), ns)
            figs = [ns[variable_name]]

    except (Exception, SystemExit) as err:
        raise PlotError(traceback.format_exc()) from err
    finally:
        os.chdir(pwd)

    return figs


def get_plot_formats(config):
    default_dpi = {"html": 0}
    formats = []
    plot_formats = config.plotly_formats
    for fmt in plot_formats:
        if isinstance(fmt, str):
            if ":" in fmt:
                suffix, dpi = fmt.split(":")
                formats.append((str(suffix), int(dpi)))
            else:
                formats.append((fmt, default_dpi.get(fmt, 80)))
        elif isinstance(fmt, (tuple, list)) and len(fmt) == 2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise PlotError('invalid image format "%r" in plot_formats' % fmt)
    return formats


def render_figures(
    code,
    code_path,
    output_dir,
    output_base,
    context,
    function_name,
    config,
    context_reset=False,
    close_figs=False,
    fig_vars=None,
):
    """
    Run a pyplot script and save the images in *output_dir*.

    Save the images under *output_dir* with file names derived from
    *output_base*
    """
    formats = get_plot_formats(config)

    # -- Try to determine if all images already exist

    code_pieces = split_code_at_show(code)

    # Look for single-figure output files first
    all_exists = True
    fig = FigureFile(output_base, output_dir)
    for format, dpi in formats:
        if out_of_date(code_path, fig.filename(format)):
            all_exists = False
            break
        fig.formats.append(format)

    if all_exists:
        return [(code, [fig])]

    # Then look for multi-figure output files
    results = []
    all_exists = True
    for i, code_piece in enumerate(code_pieces):
        figures = []
        for j in itertools.count():
            if len(code_pieces) > 1:
                fig = FigureFile("%s_%02d_%02d" % (output_base, i, j), output_dir)
            else:
                fig = FigureFile("%s_%02d" % (output_base, j), output_dir)
            for fmt, dpi in formats:
                if out_of_date(code_path, fig.filename(fmt)):
                    all_exists = False
                    break
                fig.formats.append(fmt)

            # assume that if we have one, we have them all
            if not all_exists:
                all_exists = j > 0
                break
            figures.append(fig)
        if not all_exists:
            break
        results.append((code_piece, figures))

    if all_exists:
        return results

    # We didn't find the files, so build them

    results = []
    if context:
        ns = plot_context
    else:
        ns = {}

    if context_reset:
        plot_context.clear()

    close_figs = not context or close_figs

    for i, code_piece in enumerate(code_pieces):

        if not context:
            pass
        elif close_figs:
            pass

        fig_objects = run_code(code_piece, code_path, ns, function_name, fig_vars)

        figures = []
        for j, fig_obj in enumerate(fig_objects):
            if len(fig_objects) == 1 and len(code_pieces) == 1:
                fig = FigureFile(output_base, output_dir)
            elif len(code_pieces) == 1:
                fig = FigureFile("%s_%02d" % (output_base, j), output_dir)
            else:
                fig = FigureFile("%s_%02d_%02d" % (output_base, i, j), output_dir)
            figures.append(fig)
            for fmt, dpi in formats:
                try:
                    save_plotly_figure(fig_obj, fig.filename(fmt))
                except Exception as err:
                    raise PlotError(traceback.format_exc()) from err
                fig.formats.append(fmt)

        results.append((code_piece, figures))

    if not context:
        pass

    return results


def run(arguments, content, options, state_machine, state, lineno):
    document = state_machine.document
    config = document.settings.env.config
    nofigs = "nofigs" in options

    formats = get_plot_formats(config)
    default_fmt = formats[0][0]

    options_copy = copy.deepcopy(options)

    options.setdefault("include-source", config.plotly_include_source)
    options.setdefault("iframe-width", config.plotly_iframe_width)
    options.setdefault("iframe-height", config.plotly_iframe_height)
    keep_context = "context" in options
    context_opt = None if not keep_context else options["context"]

    rst_file = document.attributes["source"]
    rst_dir = os.path.dirname(rst_file)

    if len(arguments):
        if not config.plotly_basedir:
            source_file_name = os.path.join(setup.app.builder.srcdir, directives.uri(arguments[0]))
        else:
            source_file_name = os.path.join(
                setup.confdir, config.plotly_basedir, directives.uri(arguments[0])
            )

        # If there is content, it will be passed as a caption.
        caption = "\n".join(content)

        # Enforce unambiguous use of captions.
        if "caption" in options:
            if caption:
                raise ValueError(
                    "Caption specified in both content and options." " Please remove ambiguity."
                )
            # Use caption option
            caption = options["caption"]

        # If the optional function name is provided, use it
        if len(arguments) == 2:
            function_name = arguments[1]
        else:
            function_name = None

        code = Path(source_file_name).read_text(encoding="utf-8")
        output_base = os.path.basename(source_file_name)
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get("_plot_counter", 0) + 1
        document.attributes["_plot_counter"] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        output_base = "%s-%d.py" % (base, counter)
        function_name = None
        caption = options.get("caption", "")

    base, source_ext = os.path.splitext(output_base)
    if source_ext in (".py", ".rst", ".txt"):
        output_base = base
    else:
        source_ext = ""

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace(".", "-")

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if "format" in options:
        if options["format"] == "python":
            is_doctest = False
        else:
            is_doctest = True

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = os.path.dirname(source_rel_name)
    while source_rel_dir.startswith(os.path.sep):
        source_rel_dir = source_rel_dir[1:]

    # build_dir: where to place output files (temporarily)
    build_dir = os.path.join(
        os.path.dirname(setup.app.doctreedir), "plot_directive", source_rel_dir
    )
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # output_dir: final location in the builder's directory
    dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir, source_rel_dir))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # no problem here for me, but just use built-ins

    # how to link to files from the RST file
    dest_dir_link = os.path.join(relpath(setup.confdir, rst_dir), source_rel_dir).replace(
        os.path.sep, "/"
    )
    try:
        build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, "/")
    except ValueError:
        # on Windows, relpath raises ValueError when path and start are on
        # different mounts/drives
        build_dir_link = build_dir
    source_link = dest_dir_link + "/" + output_base + source_ext

    # make figures
    try:
        results = render_figures(
            code,
            source_file_name,
            build_dir,
            output_base,
            keep_context,
            function_name,
            config,
            context_reset=context_opt == "reset",
            close_figs=context_opt == "close-figs",
            fig_vars=options.get("fig-vars"),
        )
        errors = []
    except PlotError as err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2,
            "Exception occurred in plotting {}\n from {}:\n{}".format(
                output_base, source_file_name, err
            ),
            line=lineno,
        )
        results = [(code, [])]
        errors = [sm]

    # Properly indent the caption
    caption = "\n".join("      " + line.strip() for line in caption.split("\n"))

    # generate output restructuredtext
    total_lines = []
    for j, (code_piece, figures) in enumerate(results):
        if options["include-source"]:
            if is_doctest:
                lines = ["", *code_piece.splitlines()]
            else:
                lines = [
                    ".. code-block:: python",
                    "",
                    *textwrap.indent(code_piece, "    ").splitlines(),
                ]
            source_code = "\n".join(lines)
        else:
            source_code = ""

        if nofigs:
            figures = []

        opts = [
            ":%s: %s" % (key, val)
            for key, val in options.items()
            if key in ("alt", "height", "width", "scale", "align", "class")
        ]

        # Not-None src_link signals the need for a source link in the generated
        # html
        if j == 0 and config.plotly_html_show_source_link:
            src_link = source_link
        else:
            src_link = None

        if config.plotly_include_directive_source:
            directive_source = create_directive_block("plotly", arguments, options_copy, content)
            directive_source = create_code_block(directive_source, "text")
        else:
            directive_source = ""

        result = jinja2.Template(config.plotly_template or TEMPLATE).render(
            directive_source=directive_source,
            default_fmt=default_fmt,
            dest_dir=dest_dir_link,
            build_dir=build_dir_link,
            source_link=src_link,
            multi_figure=len(figures) > 1,
            options=opts,
            figures=figures,
            iframe_width=options["iframe-width"],
            iframe_height=options["iframe-height"],
            source_code=source_code,
            html_show_formats=config.plotly_html_show_formats and len(figures),
            caption=caption,
        )

        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # copy image files to builder's output directory, if necessary
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for code_piece, figures in results:
        for fig in figures:
            for fn in fig.filenames():
                destfig = os.path.join(dest_dir, os.path.basename(fn))
                if fn != destfig:
                    shutil.copyfile(fn, destfig)

    # copy script (if necessary)
    Path(dest_dir, output_base + source_ext).write_text(
        unescape_doctest(code) if source_file_name == rst_file else code,
        encoding="utf-8",
    )

    return errors