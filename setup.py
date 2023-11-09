"""
pycatch

setup file

@author: S.G. Heinemann
"""

from setuptools import setup, find_packages
import pathlib


version = {}
with open("pycatch/_version.py") as version_file:
    exec(version_file.read(), version)

DESCRIPTION = 'Collection of Analysis Tools for Coronal Holes'
here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pycatch", 
        version=version['__version__'],
        author="Stephan G. Heinemann",
        author_email="stephan.heinemann@hmail.at",
        description=DESCRIPTION,
        long_description=long_description,
        url="https://github.com/sgheinemann/pycatch",
        packages=find_packages(),
        python_requires=">=3.11, <4",
        install_requires=['numpy','sunpy','astropy','opencv-python','matplotlib', 'reproject','scipy','numexpr','joblib','aiapy' ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: Alpha",
            "Programming Language :: Python :: 3",
]
)
