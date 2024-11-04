from setuptools import setup, find_packages

setup(
    name="HomologyRing",
    version="0.1",
    author="Tanner Aaron Graves",
    author_email="tanner.graves1@gmail.com",
    packages=find_packages(), 
    entry_points={
        'console_scripts': [
            'ring-homology=pipeline.HomologyRing:main',  # Create a command-line script
        ],
    },
    install_requires=[
        "pandas>=1.1",
        "numpy>=1.19",
        "matplotlib>=3.3",
        "requests",
        "plotly>=4.14",
        "scipy>=1.5",
        "networkx>=2.5",
        "biopython>=1.78",
        "pdbecif>=0.2.0"
    ],
    python_requires='>=3.6',
    include_package_data=True,

)
