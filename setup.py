from setuptools import setup

setup(
    name="stnn",
    version="0.0.1",    
    description="A packaged version of Delasalles et al.'s stnn model",
    url="https://github.com/nardus/stnn",
    author="Nardus Mollentze and Edouard Delasalles",
    packages=['stnn'],
    install_requires=["matplotlib>=3.4.2",
                      "numpy>=1.21.0",
                      "torch>=1.9.0",
                      "configargparse>=1.5.1",
                      "tqdm>=4.61.1",
                      "ray>=1.6.0"],

    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",  
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
