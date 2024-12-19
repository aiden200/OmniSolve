from setuptools import setup, find_packages

setup(
    name="OmniSolve",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add dependencies here, for example:
        "moviepy",
        "numpy",
        "torch",
    ],
    python_requires=">=3.10",
)
