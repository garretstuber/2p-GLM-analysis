from setuptools import setup, find_packages

setup(
    name="glm_permutation",
    version="0.1.0",
    description="Circular-shift permutation GLM for 2-photon calcium imaging",
    author="Garret Stuber Lab",
    author_email="gstuber@uw.edu",
    url="https://github.com/garretstuber/2p-GLM-analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "h5py>=3.0",
        "matplotlib>=3.5",
    ],
    extras_require={
        "notebook": ["jupyter", "ipywidgets"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
