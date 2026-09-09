
from setuptools import setup

setup(
    name="STS-Annotator",
    version="1.0.0",
    description="Semantic Similarity Data Annotator",
    author="Your Name",
    py_modules=["sts_annotator"],
    install_requires=[
        "pandas>=1.3.0",
        "PyQt5>=5.15.0",
    ],
    entry_points={
        "console_scripts": [
            "sts-annotator=sts_annotator:main",
        ],
    },
)