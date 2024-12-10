### Multi-Scale Contrastive Learning with Hierarchical Knowledge Synergy for Cross-Modality Camera Person Re-identification

Pytorch Code of MCLNet for VI-ReID on SYSU-MM01, LLCM, and RegDB datasets. 

<img src="./materials/model.png" alt="model" style="zoom: 50%;" />

#### 1. Prepare the datasets.

- RegDB Dataset: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
- SYSU-MM01 Dataset: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).
- LLCM Dataset: The LLCM dataset can be downloaded from this  [website](https://github.com/ZYK100/LLCM/blob/main/LLCM%20Dataset%20Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf)

#### 2. Training.

Train a model by

```bash
python main_train.py
```

*Hyperparameter settings*:  `config/baseline.yaml`. 

#### 3. Testing.

```bash
python main_test.py --resume --resume_path 'model_path'
```

* `--resume`: resume from checkpoint.
* `--resume_path`: model path.

*Hyperparameter settings*:  `config/baseline.yaml`. 

#### 4. References.

```bash
[1]Ye, Mang and Shen, Jianbing and Lin, Gaojie and Xiang, Tao and Shao, Ling and Hoi, Steven CH. Deep learning for person re-identification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence, vol.44, pp.2872--2893, 2021
```

```bash
[2]Jambigi, Chaitra and Rawal, Ruchit and Chakraborty, Anirban.Mmd-reid: A simple but effective solution for visible-thermal person reid. arXiv preprint arXiv:2111.05059, 2021.
```

```bash
[3]Qian, Yongheng and Tang, Su-Kit. Pose Attention-Guided Paired-Images Generation for Visible-Infrared Person Re-Identification.IEEE Signal Processing Letters, vol.31, pp.346--350, 2024.
```

