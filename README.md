SPOC Active Learning
============================

## Setup

#### Requirements
  * Python == 3.8
  * conda (recommended)

#### Install dependencies

If you want to use data loader by installing PyTorch, please check `install.sh`.
The dependencies can be installed using the following command (conda required):
```bash
bash install.sh
```

If you simply want to use data api, please check `requirements.txt`.
```bash
pip install -r requirements.txt
```


## Run

You can run the experiments with the command below.
```bash
python run.py --config_path ./configs/polymer_rf.yaml --tag TEST
```

Sample configurations can be found in the <a href='./configs'>configs</a> directory.

**NOTE**: You do not need to specify the training directory.<br>
If run as above, a unique identifiable directory will be created in `/usr/workspace/$USER` with `$TAG`.


## Acknowledgment
Many of core functions are borrowed from [this repo](https://github.com/PV-Lab/Benchmarking).

## Citation
```BibTex
@article{schwartz2025spoc,
  title={{Studying-Polymers-On a-Chip (SPOC): High-Throughput Screening of Polymers for Battery Applications}},
  author={Schwartz, Johanna and
          Jimenez, Jayvic and
          Marufu, Michell and
          Silverman, Micah and
          Tzintzun, Santiago and
          Au, Brian and
          Cerda, Robert and
          Igtanloc, Chelsea and
          Rivadeneira Velasco, Katherine and
          Elshatoury, Maged and
          Lau, Chloe and
          Hu, Zeyuan and
          Ojal, Nishant and
          Wood, Marissa and
          Xiao, Yiran and
          Cho, Seongkoo and
          Gongora, Aldair and
          Noh, Junhyug and
          Massey, Travis and
          Marple, Maxwell},
  year={2025}
}
```
