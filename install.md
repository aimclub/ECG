## How to install
* [ECG Library](index.md)
    * [Project structure](project_structure.md)
    * [Implemented methods](methods.md)
    * [API](api.md)
    * [How to install](install.md)
    * [How to use](how-to-use.md)
    * [Benchmarks](benchmarks.md)
    * [How to contribute](how-to-contribute.md)
    * [How to cite](how-to-cite.md)
    * [Contact us](contact.md)

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
    $ python -m pip install git+ssh://git@github.com/tanyapole/ECG
    ```