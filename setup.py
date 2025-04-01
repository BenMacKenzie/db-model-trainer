from setuptools import setup, find_packages

setup(
    name="db-model-trainer",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "click>=8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'yaml-processor=db_model_trainer.cli:run_cli',
        ],
    },
    author="Ben Mackenzie",
    author_email="benmackenzie2004@yahoo.com",
    description="A package for processing YAML in Databricks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BenMacKenzie/db-model-trainer.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
) 