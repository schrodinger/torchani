# Install

TorchANI requires the latest preview version of PyTorch. Please install PyTorch before installing TorchANI.

Please see [PyTorch's official site](https://pytorch.org/get-started/locally/) for instructions of installing latest preview version of PyTorch.

Note that if you updated TorchANI, you may also need to update PyTorch.

After installing the correct PyTorch, you can install TorchANI by `pip` or `conda`:

```bash
pip install torchani
```

or

```bash
conda install -c conda-forge torchani
```

See https://github.com/conda-forge/torchani-feedstock for more information about the conda package.

To run the tests and examples, you must manually download a data package

```bash
./download.sh
```

(Optional) To install AEV CUDA Extension (speedup for AEV computation), please follow the instruction at [torchani/cuaev](https://github.com/aiqm/torchani/tree/master/torchani/cuaev).

# Citation

Please cite the following paper if you use TorchANI 

* Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg. *TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials*. Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415, [![DOI for Citing](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.0c00451-green.svg)](https://doi.org/10.1021/acs.jcim.0c00451)

[![JCIM Cover](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jcisd8/2020/jcisd8.2020.60.issue-7/jcisd8.2020.60.issue-7/20200727/jcisd8.2020.60.issue-7.largecover.jpg)](https://pubs.acs.org/toc/jcisd8/60/7)

* Please refer to [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI) for ANI model references.

# ANI model parameters
All the ANI model parameters including (ANI2x, ANI1x, and ANI1ccx) are accessible from the following repositories:
- [isayev/ASE_ANI](https://github.com/isayev/ASE_ANI)
- [aiqm/ani-model-zoo](https://github.com/aiqm/ani-model-zoo)


# Develop

To install TorchANI from GitHub:

```bash
git clone https://github.com/aiqm/torchani.git
cd torchani
pip install -e .
```

After TorchANI has been installed, you can build the documents by running `sphinx-build docs build`. But make sure you
install dependencies:
```bash
pip install -r docs_requirements.txt
```

To manually run unit tests, do

```bash
pytest -v
```

If you opened a pull request, you could see your generated documents at https://aiqm.github.io/torchani-test-docs/ after you `docs` check succeed.
Keep in mind that this repository is only for the purpose of convenience of development, and only keeps the latest push.
The CI runing for other pull requests might overwrite this repository. You could rerun the `docs` check to overwrite this repo to your build.


# Note to TorchANI developers

Never commit to the master branch directly. If you need to change something, create a new branch, submit a PR on GitHub.

You must pass all the tests on GitHub before your PR can be merged.

Code review is required before merging pull request.
