#!/usr/bin/env python3
"""
ROCKETSHIP: A flexible and modular software tool for the planning, 
processing and analysis of dynamic MRI studies.

Python port of the original MATLAB version.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rocketship",
    version="2.0.0",
    author="Thomas Ng, Samuel Barnes",
    author_email="thomasn@caltech.edu, srbarnes@caltech.edu",
    description="Processing and analysis of dynamic MRI studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petmri/ROCKETSHIP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rocketship-dce=rocketship.cli:dce_main",
            "rocketship-parametric=rocketship.cli:parametric_main", 
            "rocketship-analysis=rocketship.cli:analysis_main",
        ],
    },
    include_package_data=True,
    package_data={
        "rocketship": ["config/*.txt", "config/*.cfg"],
    },
)