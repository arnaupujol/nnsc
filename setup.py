import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="nnsc",
    version="1.0.0",
    author="arnaupujol",
    author_email="arnaupv@gmail.com",
    description="Neural Network for Shear Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnaupujol/nnsc",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
