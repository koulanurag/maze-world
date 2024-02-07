from os import path

import setuptools
from setuptools import setup

extras = {
    "test": ["pytest<8.0.0", "pytest_cases"],
    "develop": ["black", "imageio", "jupyter"],
    "docs": [
        "Sphinx<6.0,>=4.0",
        "furo",
        "sphinxcontrib-katex",
        "sphinx-copybutton",
        "sphinx_design",
        "myst-parser",
        "sphinx-autobuild",
        "sphinxext-opengraph",
        "sphinx-prompt",
        "sphinx-favicon",
        "nbsphinx>=0.9.3",
        "pandoc>=2.3",
        "myst-nb",
    ],
}

# Meta dependency groups.
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="maze-world",
    version="0.0.1",
    description="A collection of maze navigation tasks.",
    long_description_content_type="text/markdown",
    long_description=open(
        path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
    ).read(),
    url="https://github.com/koulanurag/maze-world",
    author="Anurag Koul",
    author_email="koulanurag@gmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=["gymnasium>=0.28", "pygame>=2.1.0"],
    extras_require=extras,
    tests_require=extras["test"],
    python_requires=">=3.6, <3.12",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
