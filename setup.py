from setuptools import setup


with open("README.md") as f:
    README = f.read()

# This call to setup() does all the work
setup(
    name="racebert",
    version="1.0.0",
    description="Race and Ethnicity Prediction from names",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/parasurama/raceBERT",
    author="Prasanna Parasurama",
    author_email="pparasurama@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["racebert"],
    install_requires=["transformers"],
)
