
# CBMAD: Consistent Bidirectional Mamba Autoencoder for Time-Series Anomaly Detection

This repository provides an implementation of **CBMAD (Consistency Bidirectional Mamba Autoencoder Anomaly Detection)** model for unsupervised anomaly detection in multivariate time-series data.  

It is built on top of the [DeepOD](https://github.com/xuhongzuo/DeepOD) framework with additional model architectural optimizations, while retaining the core principle of leveraging **bidirectional representation learning with consistency regularization** for robust anomaly detection.


📄 **Reference Paper**: [CBMAD: Anomaly Detection in IoT Network Traffic via Consistent Bidirectional Mamba Autoencoder](https://ieeexplore.ieee.org/abstract/document/11038914)


---


## 🔧 Installation

This implementation has been tested with **PyTorch 2.4.1** and **CUDA 12.4**.

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
````
To install the required dependencies and set up the environment, run the following commands:

```bash
git clone https://github.com/YYZStart/CBMAD.git
cd CBMAD
pip install -r requirements.txt
````

---

## 🚀 Usage

CBMAD can be used for both **training + inference** and **inference only**.  
You need to specify the dataset and its corresponding feature dimension (`--nb_feature`).  

For example, for the **SMD** dataset (with 38 features):  


### Train and Inference
```bash
python main.py --dataset SMD --nb_feature 38 --action train_and_infer
```

This will train a CBMAD model on the SMD dataset and automatically run inference afterwards.

The trained model checkpoint will be saved under ./checkpoints/.

### Inference Only

If you already have a trained model checkpoint, you can directly run:

```bash
python main.py --dataset SMD --nb_feature 38 --action infer
```

This will load the saved model and perform anomaly detection on the test set.


## 📊 Datasets

CBMAD is designed to process multivariate time-series datasets.

Put the dataset in the `Dataset/` directory.

If you would like to use your datasets, please refer to **preprocessing.py** for data formatting and processing guidelines.



---



## 🧩 Plug CBMAD into Your Own Framework

If you'd like to use **CBMAD** in another training or evaluation pipeline, you can easily reuse the model architecture defined in [`CBMAD_model.py`](./model/CBMAD_model.py).

You can follow the training and inference procedure provided in this repository, or integrate CBMAD into your own training loop, datasets, or evaluation pipeline as needed.

---


## 📎 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{yu2025cbmad,
  title={CBMAD: Anomaly Detection in IoT Network Traffic via Consistent Bidirectional Mamba Autoencoder},
  author={Yu, Yuan-Cheng and Ouyang, Yen-Chieh and Lin, Chun-An},
  booktitle={2025 IEEE 26th International Conference on High Performance Switching and Routing (HPSR)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```


## 🙏 Acknowledgement

The detection framework of this project is built upon by the following repositories:

https://github.com/xuhongzuo/DeepOD

We sincerely thank the authors of these works for open-sourcing their code.

