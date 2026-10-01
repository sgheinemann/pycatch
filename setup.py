"""
pycatch

setup file

@author: S.G. Heinemann
"""

import pathlib
from setuptools import find_packages, setup

# Load version dynamically from pycatch/_version.py
version = {}
with open("pycatch/_version.py") as version_file:
    exec(version_file.read(), version)

DESCRIPTION = "Collection of Analysis Tools for Coronal Holes"
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pycatch",
    version=version["__version__"],
    author="Stephan G. Heinemann",
    author_email="stephan.heinemann@hmail.at",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgheinemann/pycatch",
    packages=find_packages(),
    python_requires=">=3.14",
    install_requires=[
        "numpy~=2.5.3",
        "astropy~=8.0.1",
        "sunpy~=8.0.0",
        "aiapy~=0.12.0",
        "opencv-python~=5.0.0",
        "matplotlib~=3.11.1",
        "reproject~=0.21.0",
        "scipy~=1.18.1",
        "numexpr~=2.14.2",
        "joblib~=1.6.0",
        "jupyterlab~=4.6.4", 
        "ipympl~=0.10.0",
    ],
    keywords=["python", "solar-physics", "coronal-holes", "sunpy", "astropy"],
    classifiers=[
        "Development Status :: Release",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
    ],
)