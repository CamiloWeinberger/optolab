# How to use

## 1. Install Anaconda

- Download Anaconda from [here](https://www.anaconda.com/products/individual#Downloads)

## 2. Create a new environment & activate it

```bash
$ conda create -n pyramidal python=3.9
$ conda activate pyramidal
```

## 3. Install pytorch

- Install pytorch from [here](https://pytorch.org/get-started/locally/)
- If you have a GPU, install pytorch with GPU support
- If you don't have a GPU, install pytorch with CPU support

- Install latest with CUDA 11.3

```bash
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## 4. Install requirements

```bash
$ conda install -c anaconda matplotlib ipykernel pandas
$ pip install seaborn pandas scikit-learn scikit-image scipy pytorch-lightning mlflow deepspeed
```

## 5. Compile server

```bash
$ python compile_server.py
```

## 6. Install the package

```bash
$ pip install -e .
```
