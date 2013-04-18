import os
from os.path import splitext, sep as filesep, join as pjoin, relpath
from hashlib import sha1
from subprocess import check_call

from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist
from distutils.version import LooseVersion


def derror_maker(klass, msg):
    """ Decorate distutils class to make run method raise error """
    class K(klass):
        def run(self):
            raise RuntimeError(msg)
    return K


def stamped_pyx_ok(exts, hash_stamp_fname):
    """ Check for match of recorded hashes for pyx, corresponding c files

    Parameters
    ----------
    exts : sequence of ``Extension``
        distutils ``Extension`` instances, in fact only need to contain a
        ``sources`` sequence field.
    hash_stamp_fname : str
        filename of text file containing hash stamps

    Returns
    -------
    tf : bool
        True if there is a corresponding c file for each pyx or py file in
        `exts` sources, and the hash for both the (pyx, py) file *and* the c
        file match those recorded in the file named in `hash_stamp_fname`.
    """
    # Calculate hashes for pyx and c files.  Check for presence of c files.
    stamps = {}
    for mod in exts:
        for source in mod.sources:
            base, ext = splitext(source)
            if not ext in ('.pyx', '.py'):
                continue
            source_hash = sha1(open(source, 'rb').read()).hexdigest()
            c_fname = base + '.c'
            try:
                c_file = open(c_fname, 'rb')
            except IOError:
                return False
            c_hash = sha1(c_file.read()).hexdigest()
            stamps[source_hash] = source
            stamps[c_hash] = c_fname
    # Read stamps from hash_stamp_fname; check in stamps dictionary
    try:
        stamp_file = open(hash_stamp_fname, 'rt')
    except IOError:
        return False
    for line in stamp_file:
        if line.startswith('#'):
            continue
        fname, hash = [e.strip() for e in line.split(',')]
        if not hash in stamps:
            return False
        # Compare path made canonical for \/
        fname = fname.replace(filesep, '/')
        if not stamps[hash].replace(filesep, '/') == fname:
            return False
        stamps.pop(hash)
    # All good if we found all hashes we need
    return len(stamps) == 0


def cyproc_exts(exts, cython_min_version,
                hash_stamps_fname = 'pyx-stamps',
                build_ext=build_ext):
    """ Process sequence of `exts` to check if we need Cython.  Return builder

    Parameters
    ----------
    exts : sequence of distutils ``Extension``
        If we already have good c files for any pyx or py sources, we replace
        the pyx or py files with their compiled up c versions inplace.
    cython_min_version : str
        Minimum cython version neede for compile
    hash_stamps_fname : str, optional
        filename with hashes for pyx/py and c files known to be in sync. Default
        is 'pyx-stamps'
    build_ext : distutils command
        default build_ext to return if not cythonizing.  Default is distutils
        ``build_ext`` class

    Returns
    -------
    builder : ``distutils`` ``build_ext`` class or similar
        Can be ``build_ext`` input (if we have good c files) or cython
        ``build_ext`` if we have a good cython, or a class raising an informative
        error on ``run()``
    """
    if stamped_pyx_ok(exts, hash_stamps_fname):
        # Replace pyx with c files, use standard builder
        for mod in exts:
            sources = []
            for source in mod.sources:
                base, ext = splitext(source)
                if ext in ('.pyx', '.py'):
                    sources.append(base + '.c')
                else:
                    sources.append(source)
            mod.sources = sources
        return build_ext
    # We need cython
    try:
        from Cython.Compiler.Version import version as cyversion
    except ImportError:
        cython_ok = False
    else:
        cython_ok = LooseVersion(cyversion) >= cython_min_version
    if cython_ok:
        from Cython.Distutils import build_ext as extbuilder
        return extbuilder
    return derror_maker(build_ext,
                        'Need cython>=%s to build extensions'
                        % cython_min_version)


def build_stamp(pyxes):
    """ Cythonize files in `pyxes`, return pyx, C filenames, hashes

    Parameters
    ----------
    pyxes : sequence
        sequence of filenames of files on which to run Cython

    Returns
    -------
    pyx_defs : dict
        dict has key, value pairs of <pyx_filename>, <pyx_info>, where
        <pyx_info> is a dict with key, value pairs of "pyx_hash", <pyx file SHA1
        hash>; "c_filename", <c filemane>; "c_hash", <c file SHA1 hash>.
    """
    pyx_defs = {}
    for source in pyxes:
        base, ext = splitext(source)
        pyx_hash = sha1(open(source, 'rt').read()).hexdigest()
        c_filename = base + '.c'
        check_call('cython ' + source, shell=True)
        c_hash = sha1(open(c_filename, 'rt').read()).hexdigest()
        pyx_defs[source] = dict(pyx_hash=pyx_hash,
                                c_filename=c_filename,
                                c_hash=c_hash)
    return pyx_defs


def write_stamps(pyx_defs, stamp_fname='pyx-stamps'):
    """ Write stamp information in `pyx_defs` to filename `stamp_fname`

    Parameters
    ----------
    pyx_defs : dict
        dict has key, value pairs of <pyx_filename>, <pyx_info>, where
        <pyx_info> is a dict with key, value pairs of "pyx_hash", <pyx file SHA1
        hash>; "c_filename", <c filemane>; "c_hash", <c file SHA1 hash>.
    stamp_fname : str
        filename to which to write stamp information
    """
    with open(stamp_fname, 'wt') as stamp_file:
        stamp_file.write('# SHA1 hashes for pyx files and generated c files\n')
        stamp_file.write('# Auto-generated file, do not edit\n')
        for pyx_fname, pyx_info in pyx_defs.items():
            stamp_file.write('%s, %s\n' % (pyx_fname,
                                           pyx_info['pyx_hash']))
            stamp_file.write('%s, %s\n' % (pyx_info['c_filename'],
                                           pyx_info['c_hash']))


def find_pyx(root_dir=None):
    """ Recursively find files with extension '.pyx' starting at `root_dir`

    Parameters
    ----------
    root_dir : None or str, optional
        Directory from which to search for pyx files.  If None, use current
        working directory.

    Returns
    -------
    pyxes : list
        list of filenames relative to `root_dir`
    """
    if root_dir is None:
        root_dir = os.getcwd()
    pyxes = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.pyx'):
                continue
            base = relpath(dirpath, root_dir)
            pyxes.append(pjoin(base, filename))
    return pyxes


def get_pyx_sdist(sdist_like=sdist, hash_stamps_fname='pyx-stamps'):
    """ Add pyx->c conversion, hash recording to sdist command `sdist_like`

    Parameters
    ----------
    sdist_like : sdist command class, optional
        command that will do work of ``distutils.command.sdist.sdist``.  By
        default we use the distutils version
    hash_stamps_fname : str, optional
        filename to which to write hashes of pyx / py and c files.  Default is
        ``pyx-stamps``

    Returns
    -------
    modified_sdist : sdist-like command class
        decorated `sdist_like` class, for compiling pyx / py files to c, putting
        the .c files in the the source archive, and writing hashes for these
        into the file named from `hash_stamps_fname`
    """
    class PyxSDist(sdist_like):
        """ Custom distutils sdist command to generate .c files from pyx files.

        Running the command object ``obj.run()`` will compile the pyx / py files
        in any extensions, into c files, and add them to the list of files to
        put into the source archive, as well as the usual behavior of distutils
        ``sdist``.  It will also take the sha1 hashes of the pyx / py and c
        files, and store them in a file ``pyx-stamps``, and put this file in the
        release tree.  This allows someone who has the archive to know that the
        pyx and c files that they have are the ones packed into the archive, and
        therefore they may not need Cython at install time.  See
        ``cython_process_exts`` for the build-time command.
        """

        def make_distribution(self):
            """ Compile pyx to c files, add to sources, stamp sha1s """
            pyxes = []
            for mod in self.distribution.ext_modules:
                for source in mod.sources:
                    base, ext = splitext(source)
                    if ext in ('.pyx', '.py'):
                        pyxes.append(source)
            self.pyx_defs = build_stamp(pyxes)
            for pyx_fname, pyx_info in self.pyx_defs.items():
                self.filelist.append(pyx_info['c_filename'])
            sdist_like.make_distribution(self)

        def make_release_tree(self, base_dir, files):
            """ Put pyx stamps file into release tree """
            sdist_like.make_release_tree(self, base_dir, files)
            stamp_fname = pjoin(base_dir, hash_stamps_fname)
            write_stamps(self.pyx_defs, stamp_fname)

    return PyxSDist


def build_stamp_source(root_dir=None, stamp_fname='pyx-stamps'):
    """ Build cython c files, make stamp file in source tree `root_dir`

    Parameters
    ----------
    root_dir : None or str, optional
        Directory from which to find ``.pyx`` files.  If None, use current
        working directory.
    stamp_fname : str, optional
        Filename for stamp file we will write
    """
    pyxes = find_pyx(root_dir)
    pyx_defs = build_stamp(pyxes)
    write_stamps(pyx_defs, stamp_fname)
