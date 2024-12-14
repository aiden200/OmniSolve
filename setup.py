from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyYAML",
        "torch",
        "transformers",
        # Add more dependencies here
    ],
)

