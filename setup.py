import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuroai",
    version="0.0.0",
    description="tools for the neuroAI course",
    author="MurtyLab @ Georgia Tech",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/murtylab/neuroai-notebooks",
    packages=setuptools.find_packages(),
    install_requires=None,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)