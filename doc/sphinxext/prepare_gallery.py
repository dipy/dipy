from dataclasses import dataclass, field
import fnmatch
import os
from os.path import join as pjoin
from pathlib import Path
import shutil
import sys

from sphinx.util import logging

try:
    import tomllib
except ImportError:
    import tomli as tomllib

logger = logging.getLogger(__name__)

sphx_glr_sep = '########################################'
sphx_glr_sep += '#######################################'
sphx_glr_sep_2 = '#%%'


@dataclass(order=True)
class examplesConfig:
    sort_index: int = field(init=False)
    readme: str
    position: int
    enable: bool
    files: list
    folder_name: str

    def __post_init__(self):
        self.sort_index = self.position


def abort(error):
    print(f'*WARNING* Examples Revamp not generated: \n\n{error}')
    exit()


def already_converted(content):
    """Check if the example file is already converted to sphinx-gallery format.

    Parameters
    ----------
    content : list of str
        example file content

    Returns
    -------
    bool
        _description_
    """
    return content.count(sphx_glr_sep) >= 2 or \
        content.count(sphx_glr_sep_2) >= 2


def convert_to_sphinx_gallery_format(content):
    """Convert the example file to sphinx-gallery format.

    Parameters
    ----------
    content : list of str
        example file content

    Returns
    -------
    list of str
        example file content converted to sphinx-gallery format
    """
    inheader = True
    indocs = False

    new_content = ''
    for line in content:

        if inheader:
            if not indocs and (line.startswith('"""') or line.startswith("'''") or
               line.startswith('r"""') or line.startswith("r'''")):
                new_content += line
                if len(line.rstrip()) < 5:
                    indocs = True
                else:
                    # single line doc
                    inheader = False
                continue

            if line.rstrip().endswith('"""') or line.rstrip().endswith("'''"):
                inheader = False
                indocs = False

            new_content += line
            continue

        if indocs \
           or (line.startswith('"""') or line.startswith("'''") or
               line.startswith('r"""') or line.startswith("r'''")):

            if not indocs:
                # guaranteed to start with """
                if len(line.rstrip()) > 4 \
                  and (line.rstrip().endswith('"""') or line.rstrip().endswith("'''")):
                    # single line doc
                    tmp_line = line.replace('"""', '').replace("'''", '')
                    new_content += f'{sphx_glr_sep}\n'
                    new_content += f'# {tmp_line}'
                else:
                    # must be start of multiline block
                    indocs = True
                    new_content += f'{sphx_glr_sep}\n'
            else:
                # we are already in the docs
                # handle doc end
                if line.rstrip().endswith('"""') or line.rstrip().endswith("'''"):
                    # remove quotes
                    # import ipdb; ipdb.set_trace()
                    indocs = False
                    new_content += '\n'
                else:
                    # import ipdb; ipdb.set_trace()
                    new_content += f'# {line}'
                    # has to be documentation
            continue

        new_content += line

    return new_content


def folder_explicit_order():
    srcdir = os.path.abspath(os.path.dirname(__file__))
    examples_dir = os.path.join(srcdir, '..', 'examples')

    f_example_desc = Path(examples_dir, '_valid_examples.toml')
    if not f_example_desc.exists():
        msg = f'No valid examples description file found in {examples_dir}'
        msg += "(e.g '_valid_examples.toml')"
        abort(msg)

    with open(f_example_desc, 'rb') as fobj:
        try:
            desc_examples = tomllib.load(fobj)
        except Exception as e:
            msg = f'Error Loading examples description file: {e}.\n\n'
            msg += 'Please check the file format.'
            abort(msg)

    if 'main' not in desc_examples.keys():
        msg = 'No main section found in examples description file'
        abort(msg)

    try:
        folder_list = sorted(
            [examplesConfig(folder_name=k.lower(), **v)
             for k, v in desc_examples.items()]
        )
    except Exception as e:
        msg = f'Error parsing examples description file: {e}.\n\n'
        msg += 'Please check the file format.'
        abort(msg)

    return [f.folder_name for f in folder_list if f.enable]


def preprocess_include_directive(input_rst, output_rst):
    """
    Process include directives from input RST, and write the output to a new RST.

    Parameters
    ----------
    input_rst : str
        Path to the input RST file containing the include directive.
    output_rst : str
        Path to the output RST file with the include content expanded.

    """
    with open(input_rst, "r") as infile, open(output_rst, "w") as outfile:
        for line in infile:
            if line.strip().startswith(".. include::"):
                # Extract the included file's path
                included_file_path = line.strip().split(" ")[-1]
                included_file_path = os.path.normpath(
                    os.path.join(os.path.dirname(input_rst), included_file_path)
                )

                # Check if the included file exists and read its content
                if os.path.isfile(included_file_path):
                    with open(included_file_path, "r") as included_file:
                        included_content = included_file.read()
                        # Write the included file's content to the output file
                        outfile.write(included_content)
                else:
                    print(f"Warning: Included file '{included_file_path}' not found.")
            else:
                # Write the line as-is if it's not an include directive
                outfile.write(line)


def prepare_gallery(app=None):
    srcdir = app.srcdir if app else os.path.abspath(pjoin(
        os.path.dirname(__file__), '..'))
    examples_dir = os.path.join(srcdir, 'examples')
    examples_revamp_dir = os.path.join(srcdir, 'examples_revamped')
    os.makedirs(examples_revamp_dir, exist_ok=True)

    f_example_desc = Path(examples_dir, '_valid_examples.toml')
    if not f_example_desc.exists():
        msg = f'No valid examples description file found in {examples_dir}'
        msg += "(e.g '_valid_examples.toml')"
        abort(msg)

    with open(f_example_desc, 'rb') as fobj:
        try:
            desc_examples = tomllib.load(fobj)
        except Exception as e:
            msg = f'Error Loading examples description file: {e}.\n\n'
            msg += 'Please check the file format.'
            abort(msg)

    if 'main' not in desc_examples.keys():
        msg = 'No main section found in examples description file'
        abort(msg)

    try:
        examples_config = sorted(
            [examplesConfig(folder_name=k.lower(), **v)
             for k, v in desc_examples.items()]
        )
    except Exception as e:
        msg = f'Error parsing examples description file: {e}.\n\n'
        msg += 'Please check the file format.'
        abort(msg)

    if examples_config[0].position != 0:
        msg = 'Main section must be first in examples description file with'
        msg += 'position=0'
        abort(msg)
    elif examples_config[0].folder_name != 'main':
        msg = "Main section must be named 'main' in examples description file"
        abort(msg)
    elif examples_config[0].enable is False:
        msg = 'Main section must be enabled in examples description file'
        abort(msg)

    disable_examples_section = []
    included_examples = []
    for example in examples_config:
        if not example.enable:
            disable_examples_section.append(example.folder_name)
            continue

        # Create folder for each example
        if example.position != 0:
            folder = Path(
                examples_revamp_dir, f'{example.folder_name}')
        else:
            folder = Path(examples_revamp_dir)

        if not folder.exists():
            os.makedirs(folder)

        # Create readme file
        if example.readme.startswith('file:'):
            filename = example.readme.split('file:')[1].strip()
            preprocess_include_directive(Path(examples_dir, filename),
                                         Path(folder, "README.rst"))
        else:
            with open(Path(folder, "tmp_readme.rst"), "w", encoding="utf8") as fi:
                fi.write(example.readme)
            preprocess_include_directive(Path(folder, "tmp_readme.rst"),
                                         Path(folder, "README.rst"))
            os.remove(Path(folder, "tmp_readme.rst"))

        # Copy files to folder
        if not example.files:
            continue

        for filename in example.files:
            if not Path(examples_dir, filename).exists():
                msg = f'\tFile {filename} not found in examples folder:  '
                msg += f'{examples_dir}.Please, Add the file or remove it '
                msg += 'from the description file.'
                logger.warning(msg)
                continue

            with open(Path(examples_dir, filename), encoding="utf8") as f:
                xfile = f.readlines()

            new_name = None
            if filename in included_examples:
                # file need to be renamed to make it unique for sphinx-gallery
                occurrences = included_examples.count(fi)
                new_name = f'{filename[:-3]}_{occurrences+1}.py'
            if already_converted(xfile):
                shutil.copy(Path(examples_dir, filename),
                            Path(folder, new_name or filename))
            else:
                with open(Path(folder, new_name or filename), 'w',
                          encoding="utf8") as fi:
                    fi.write(convert_to_sphinx_gallery_format(xfile))
            # Add additional link_names
            with open(Path(folder, new_name or filename), 'r+',
                      encoding="utf8") as fi:
                content = fi.read()
                fi.seek(0, 0)
                link_name = f'\n{sphx_glr_sep}\n'
                link_name += '# .. include:: ../../links_names.inc\n#\n'
                fi.write(content + link_name)

            included_examples.append(filename)

    # Check if all python examples are in the description file
    files_in_config = [fi for ex in examples_config for fi in ex.files]
    all_examples = fnmatch.filter(os.listdir(examples_dir), '*.py')
    for all_ex in all_examples:
        if all_ex in files_in_config:
            continue
        msg = f'File {all_ex} not found in examples '
        msg += f"description file: {f_example_desc}"
        logger.warning(msg)


def setup(app):
    """Install the plugin.
    Parameters
    ----------
    app: Sphinx application context.
    """
    logger.info('Initializing Examples folder revamp plugin...')

    app.connect('builder-inited', prepare_gallery)
    # app.connect('build-finished', summarize_failing_examples)

    metadata = {'parallel_read_safe': True, 'version': app.config.version}

    return metadata


if __name__ == '__main__':
    gallery_name = sys.argv[1]
    outdir = sys.argv[2]

    print(folder_explicit_order())
    prepare_gallery(app=None)
