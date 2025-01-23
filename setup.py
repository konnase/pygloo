import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # get build dir
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={self.get_executable_path()}",
        ]
        build_args = ["--config", "Release"]

        # create build dir
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # build with cmake command
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_executable_path(self):
        return os.path.abspath(os.sys.executable)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


setup(
    name="pygloo",
    version="0.0.1",
    author="konnase123@gmail.com",
    description="A Python package for gloo",
    ext_modules=[CMakeExtension("pygloo")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)