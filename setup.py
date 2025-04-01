from setuptools import setup, find_packages

setup(
    name="db-model-trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
    ],
    author="Ben Mackenzie",
    author_email="benmackenzie2004@yahoo.com",
    description="YAML based model trainer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/db-model-trainer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 