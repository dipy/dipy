''' Distutils / setuptools helpers '''

from os.path import join as pjoin
from ConfigParser import ConfigParser

from distutils.command.build_py import build_py

def get_build_cmd(pkg_dir):
    class MyBuildPy(build_py):
        ''' Subclass to write commit data into installation tree '''
        def run(self):
            build_py.run(self)
            import subprocess
            proc = subprocess.Popen('git rev-parse --short HEAD',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=True)
            repo_commit, _ = proc.communicate()
            # We write the installation commit even if it's empty
            cfg_parser = ConfigParser()
            cfg_parser.read(pjoin(pkg_dir, 'COMMIT_INFO.txt'))
            cfg_parser.set('commit hash', 'install_hash', repo_commit)
            out_pth = pjoin(self.build_lib, pkg_dir, 'COMMIT_INFO.txt')
            cfg_parser.write(open(out_pth, 'wt'))
    return MyBuildPy


