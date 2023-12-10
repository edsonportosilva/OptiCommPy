# Authors: OpticCommPy contributors
# License: BSD 3-Clause

from setuptools import setup

# Taken from scikit-commpy setup.py
DISTNAME = "OptiCommPy"
DESCRIPTION = "Optical Communications Algorithms with Python"
LONG_DESCRIPTION = open("README.md", encoding="utf8").read()
MAINTAINER = "Edson Porto da Silva"
MAINTAINER_EMAIL = "edsonporto88@gmail.com"
URL = "https://github.com/edsonportosilva/OptiCommPy"
LICENSE = "BSD 3-Clause"
VERSION = "0.7.0"

# This is a list of files to install, and where
# (relative to the 'root' dir, where setup.py is)
# You could be more specific.
files = ["optic/*", "optic/comm/*", "optic/dsp/*", "optic/models/*"]

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    # Name the folder where your packages live:
    # (If you have other packages (dirs) or modules (py files) then
    # put them into the package directory - they will be found
    # recursively.)
    packages=["optic", "optic.comm", "optic.models", "optic.dsp"],
    # py_modules = ['plot', 'core', 'comm', 'dsp', 'models'],
    install_requires=[
        "numpy>=1.9.2",
        "scipy>=0.15.0",
        "matplotlib>=3.7.0",
        "sympy",
        "tqdm>=4.64.1",
        "numba>=0.54.1",
        "scikit-commpy>=0.7.0",
        "simple-pid>=1.0.1",
        "mpl-scatter-density>=0.7.0",
        "pandas>=2.0.0",
        "sphinx-rtd-theme>=1.2.2",
    ],
    #'package' package must contain files (see list above)
    # This dict maps the package name =to=> directories
    # It says, package *needs* these files.
    package_data={"optic": files},
    #'runner' is in the root.
    scripts=["runner"],
    test_suite="nose.collector",
    tests_require=["nose"],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    python_requires=">=3.2",
)
