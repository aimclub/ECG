![logo](docs/logo.png)

# ECG Recognition Library
This repository contains ECG Recognition Library - an open-source library for assisting in diagnostics of heart conditions from ECG. This library provides functionality of heart condition detection, differential diagnostics, and risk markers evaluation. The library handles ECGs in both formats, as signal or as a photo.

## Project Description

### ECG Features
Main features implemented in the library
1. Recognition of ECG signal from a photo of printed ECG
1. Detection of ST-elevation
1. IM risk markers evaluation
1. IM/BER differential diagnosis

Thus compared to other frameworks, ECG Recognition Library:
* Handles ECGs provised as a signal as well as an image
* Provides a range of functionality useful for IM diagnostics

![project_structure](docs/project_structure.png)

Details of [implemented methods](docs/models.md).

### Data Requirement
* Required ECG frequency: 500 Hz
* Required length: ≥ 5s

## Installation
Requirements: Python 3.7

1. [optional] create Python environment, e.g.
    ```
    $ conda create -n ECG python=3.7
    $ conda activate ECG
    ```
1. install requirements from [requirements.txt](requirements.txt)
    ```
    $ pip install -r requirements.txt
    ```
1. install the library as a package
    ```
    $ python -m pip install git+ssh://git@github.com/Med-AI-Lab/ECG
    ```

## Development
### Environment
Requirements: Python 3.7
1. [optional] create Python environment, e.g.
    ```
    $ conda create -n ECG python=3.7
    $ conda activate ECG
    ```
1. clone repository and install all requirements
    ```
    $ git clone git@github.com:Med-AI-Lab/ECG.git
    $ cd ECG
    $ pip install -r requirements.txt
    ```
1. run tests
    ```
    $ pytest tests/unit_tests.py
    $ pytest tests/integration_tests.py
    ```
1. fix code style to match PEP8 automatically with autopep8
    ```
    $ pip install autopep8==1.6.0
    $ autopep8 --max-line-length=90 -i -r ECG
    $ autopep8 --max-line-length=90 -i -r tests
    ```
1. check that code style matches PEP8
    ```
    $ pip install pycodestyle==2.8.0
    $ pycodestyle --max-line-length=90 --ignore=E266 ECG
    $ pycodestyle --max-line-length=90 --ignore=E266 tests
    ```
1. build a PyPi package locally
    ```
    $ python3 -m pip install --upgrade build
    $ python3 -m build
    ```

## Documentation
The general description is available [here](https://Med-AI-Lab.github.io/ECG/).

ECG Recognition Library API is available [here](https://med-ai-lab.github.io/ECG/docs/ECG/api.html)

## Examples & Tutorials
We provide a [tutorial](examples/intro_to_ECG.ipynb) demonstrating suggested usage pipeline

## Contribution Guide
The contribution guide is available in the [repository](./docs/contribution.md).

## Acknowledgments
We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

## Contacts
TBD

## Citation
Information will be provided after publication acceptance