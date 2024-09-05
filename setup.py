from setuptools import setup, find_packages

setup(
    name="niuload",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "transformers",
        "torch",
    ],
    author="rangehow",
    author_email="rangehow@outlook.com",
    description="A module for balanced loading of large language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rangehow/niuload",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
