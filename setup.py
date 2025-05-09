from setuptools import setup, find_packages

setup(
    name="submodular_matroids_knapsacks",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.6",
        "networkx==2.6.3",
        "matplotlib==3.4.3",
        "pandas==1.3.5",
        "scipy==1.7.3",
        "sortedcontainers==2.4.0"
    ],
)