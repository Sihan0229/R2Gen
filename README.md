# R2Gen

This is the implementation of [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf) at EMNLP-2020.

## Citations

If you use or extend our work, please cite our paper at EMNLP-2020.
```
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and
      Song, Yan  and
      Chang, Tsung-Hui and
      Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

c


## Download R2Gen
You can download the models we trained for each dataset from [here](https://github.com/cuhksz-nlp/R2Gen/blob/main/data/r2gen.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Personal changes
To do something new with the private data set, change the requirements to
`torch==1.11.0+cu113` - `torchvision==0.12.0+cu113` - `opencv-python==4.4.0`

How to set java
```
sudo apt update
sudo apt install default-jdk
sudo apt install default-jdk
vim ~/.bashrc
export JAVA_HOME=/usr/lib/jvm/default-java
export PATH=$JAVA_HOME/bin:$PATH
```
(and delete `source /etc/autodl-mot`)

finally
```
source ~/.bashrc
```

anyway, make the `.bashrc`like:
```
source /etc/profile
export JAVA_HOME=/usr/lib/jvm/default-java
export PATH=$JAVA_HOME/bin:$PATH
```