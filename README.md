# Parametric Information Maximization for Generalized Category Discovery


## Updates

## Paper
### [**Parametric Information Maximization for Generalized Category Discovery**](https://openaccess.thecvf.com/content/ICCV2023/papers/Chiaroni_Parametric_Information_Maximization_for_Generalized_Category_Discovery_ICCV_2023_paper.pdf)

If you find this code useful for your research, please cite our [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chiaroni_Parametric_Information_Maximization_for_Generalized_Category_Discovery_ICCV_2023_paper.pdf) (published at ICCV 2023):
```
@inproceedings{chiaroni2023parametric,
  title={Parametric information maximization for generalized category discovery},
  author={Chiaroni, Florent and Dolz, Jose and Masud, Ziko Imtiaz and Mitiche, Amar and Ben Ayed, Ismail},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1729--1739},
  year={2023}
}
```

### Abstract
<p align="justify">
  We introduce a Parametric Information Maximization (PIM) model for the Generalized Category Discovery (GCD) problem. Specifically, we propose a bi-level optimization formulation, which explores a parameterized family of objective functions, each evaluating a weighted mutual information between the features and the latent labels, subject to supervision constraints from the labeled samples. Our formulation mitigates the class-balance bias encoded in standard information maximization approaches, thereby handling effectively both short-tailed and long-tailed data sets. We report extensive experiments and comparisons demonstrating that our PIM model consistently sets new state-of-the-art performances in GCD across six different datasets, more so when dealing with challenging fine-grained problems. 
</p>

## Get started

### Pre-requisites
* Python 3.9.4
* numpy 1.22.0
* scikit-learn 0.24.1
* scipy 1.11.1
* yaml 6.0
* tqdm 4.65.0
* Pytorch 1.11.0 
* CUDA 11.3 

You can install all the pre-requisites using 
```bash
$ cd <root_dir>
$ pip install -r requirements.txt
```

### Feature map sets
This code enables to directly apply our approach PIM on top of the feature map sets, which can be extracted with the ViT-B-16 encoder following the training procedure proposed in [GCD code](https://github.com/sgvaze/generalized-category-discovery).
The extracted feature map sets can be directly downloaded [here](https://drive.google.com/file/d/1p0N_F-pK0Y2vKPbb7Dt3aiT8ifLtuzH6/view?usp=sharing).

### Running the code
The script pim_partitioning.py runs the proposed PIM partitioning model.

You can set the feature map set paths in the config file [`./configs/config_fm_paths.yml`](./configs/config_fm_paths.yml).

Apply PIM on a given feature map set as follows:
```bash
$ cd <root_dir>
$ python pim_partitioning.py --dataset <dataset_name>
```
where ```<dataset_name>``` must be replaced with one of the following dataset names: 
- ```cifar10``` for CIFAR-10
- ```cifar100``` for CIFAR-100
- ```imagenet_100``` for ImageNet-100
- ```cub``` for CUB
- ```scars``` for Stanford-Cars
- ```herbarium``` for Herbarium19

## Recommendations
A small lambda value close to 0 is more appropriate on balanced datasets (such as CUB) while a lambda value close to 1 is more appropriate on long-tailed imbalanced datasets (such as Herbarium19). **Note**: Our code enables, without the need of a validation set, to automatically estimate the optimal lambda value for each unlabeled feature map set.

## Contributing
If you are interested in contributing to this project, start by reading our [Contributing guide](/CONTRIBUTING.md).

## License
The code is licensed under the MIT (see [LICENSE](/LICENSE) for details).
