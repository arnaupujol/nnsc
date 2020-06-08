import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnsc",
    version="0.0.1",
    author="arnaupujol",
    author_email="arnaupv@gmail.com",
    description="Neural Network for Shear Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnaupujol/nnsc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.6',
)
