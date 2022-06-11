# ECG

## Usage
### How to install the library as a PyPi package
1. [optional] create & activate python environment, e.g. <br/>
`$ conda create -n ECG python=3.7` <br/>
`$ conda activate ECG`
1. install requirements from [requirements.txt](requirements.txt) <br/>
`$ pip install -r requirements.txt`
1. install the library as a package <br/>
`$ python -m pip install git+ssh://git@github.com/tanyapole/ECG`

## Development
### Environment
* python 3.7
* pip requirements: [requirements.txt](requirements.txt)

### How to run tests from command line
`$ pytest tests/api_tests.py`

### How to build a PyPi package locally
1. `$ python3 -m pip install --upgrade build`
2. `$ python3 -m build`


## Data Requirement
* Required ECG frequency: 500 Hz
* Required length: ≥ 5s

## Description of Models and Algorithms
[link](docs/main.md)