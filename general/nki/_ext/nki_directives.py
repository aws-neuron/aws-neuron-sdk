"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

Define new directives for nki documentation

"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Any

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.directives.code import LiteralInclude, container_wrapper, \
  LiteralIncludeReader
from sphinx.locale import __
from sphinx.util import logging, parselinenos

if TYPE_CHECKING:
  from docutils.nodes import Element, Node

  from sphinx.application import Sphinx
  from sphinx.config import Config
  from sphinx.util.typing import ExtensionMetadata, OptionSpec

logger = logging.getLogger(__name__)


class NKIExampleReader(LiteralIncludeReader):

  def __init__(self, filename: str, options: dict[str, Any], config: Config) -> None:
    if 'diff' in options:
      raise ValueError(__('`diff` mode is not supported'))

    super().__init__(filename=filename, options=options, config=config)
    marker = self.options.get('marker', 'NKI_EXAMPLE')
    self.example_begin = f'{marker}_BEGIN'
    self.example_end = f'{marker}_END'
    self.skip_marker = self.options.get('skip_marker', 'NKI_EXAMPLE')

  def nki_example_filter(
      self, lines: list[str], location: tuple[str, int] | None = None,
  ) -> list[str]:
    whole_file = 'whole-file' in self.options
    example_lines = []
    include_line = whole_file
    indentsize = 0

    for lineno, line in enumerate(lines):
      if include_line:
        if not whole_file and self.example_end in line:
          include_line = False
          continue

        if self.skip_marker in line:
          continue

        if indentsize and '\n' not in line[:indentsize]:
          line = line[indentsize:]

        example_lines.append(line)
        continue

      assert not whole_file, "`inline` should stay true if `whole_file` is True"
      if self.example_begin in line:
        include_line = True
        indentsize = len(line) - len(line.lstrip())
        if example_lines:
          # Insert an empty line between blocks
          example_lines.append('\n')

        continue

    return example_lines

  def read(self, location: tuple[str, int] | None = None) -> tuple[str, int]:
    filters = [
      self.nki_example_filter,
      #  self.pyobject_filter,
      #  self.start_filter,
      #  self.end_filter,
      #  self.lines_filter,
      self.dedent_filter,
      self.prepend_filter,
      self.append_filter]

    lines = self.read_file(self.filename, location=location)

    for func in filters:
      lines = func(lines, location=location)

    return ''.join(lines), len(lines)


class NKIExample(LiteralInclude):
  """A directive to include nki example"""
  option_spec: ClassVar[OptionSpec] = {
    'marker': str,
    'skip_marker': str,
    'whole-file': directives.flag,
    **LiteralInclude.option_spec,
  }

  def run(self) -> list[Node]:
    document = self.state.document
    if not document.settings.file_insertion_enabled:
      return [document.reporter.warning('File insertion disabled',
                                        line=self.lineno)]
    # convert options['diff'] to absolute path
    if 'diff' in self.options:
      _, path = self.env.relfn2path(self.options['diff'])
      self.options['diff'] = path

    try:
      location = self.state_machine.get_source_and_line(self.lineno)
      rel_filename, filename = self.env.relfn2path(self.arguments[0])
      self.env.note_dependency(rel_filename)

      reader = NKIExampleReader(filename, self.options, self.config)
      text, lines = reader.read(location=location)

      retnode: Element = nodes.literal_block(text, text, source=filename)
      retnode['force'] = 'force' in self.options
      self.set_source_info(retnode)
      if self.options.get('diff'):  # if diff is set, set udiff
        retnode['language'] = 'udiff'
      elif 'language' in self.options:
        retnode['language'] = self.options['language']
      if ('linenos' in self.options or 'lineno-start' in self.options or
          'lineno-match' in self.options):
        retnode['linenos'] = True
      retnode['classes'] += self.options.get('class', [])
      extra_args = retnode['highlight_args'] = {}
      if 'emphasize-lines' in self.options:
        hl_lines = parselinenos(self.options['emphasize-lines'], lines)
        if any(i >= lines for i in hl_lines):
          logger.warning(__('line number spec is out of range(1-%d): %r'),
                         lines, self.options['emphasize-lines'],
                         location=location)
        extra_args['hl_lines'] = [x + 1 for x in hl_lines if x < lines]
      extra_args['linenostart'] = reader.lineno_start

      if 'caption' in self.options:
        caption = self.options['caption'] or self.arguments[0]
        retnode = container_wrapper(self, retnode, caption)

      # retnode will be note_implicit_target that is linked from caption and numref.
      # when options['name'] is provided, it should be primary ID.
      self.add_name(retnode)

      return [retnode]
    except Exception as exc:
      return [document.reporter.warning(exc, line=self.lineno)]


def setup(app: Sphinx) -> ExtensionMetadata:
  app.add_directive('nki_example', NKIExample)

  return {
    'version': '0.1',
    'parallel_read_safe': True,
    'parallel_write_safe': True,
  }
