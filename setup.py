#!/usr/bin/env python
import os
from setuptools import find_packages, setup

from typing import List


REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def getVersion() -> str:
    """Gets version from local file"""
    with open(os.path.join(REPO_DIR, "VERSION"), "r") as versionFile:
        return versionFile.read().strip()


def getRequirements() -> List[str]:
    """Read the requirements.txt file and parse listed requirements."""
    with open(os.path.join(REPO_DIR, "requirements.txt")) as rFile:
        requirements = [r.strip() for r in rFile.readlines()]

    return requirements


if __name__ == "__main__":
    setup(name="lcml",
          version=getVersion(),
          install_requires=getRequirements(),
          packages=find_packages(),
          description="Light curve classification prototyping",
          license="MIT License",
          author="Ryan J. McCall",
          author_email="ryanjryan@protonmail.com",
          url="https://github.com/lsst-epo/light_curve_ml")
