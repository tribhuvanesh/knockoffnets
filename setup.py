import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knockoff",
    version="0.1",
    author="Tribhuvanesh Orekondy",
    author_email="orekondy@mpi-inf.mpg.de",
    description="Code for Knockoff Nets (CVPR 2019)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tribhuvanesh/knockoffnets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)