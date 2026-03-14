<p align="center">
  <h2 align="center">Leveraging neural networks and edge detection for better UAV localization 🚁🗺️📍</h2>
  <h4 align="center"><b>IGARSS 2024</b></h4>
  <p align="center">
    <a href="http://arxiv.org/pdf/2404.06207"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2404.06207-b31b1b.svg"></a>
  </p>
</p>

---

## Method Overview

Offline, the edges of the RGB reference images are extracted to generate single-channel images. An AutoEncoder is then trained on these images for a task of pixel-by-pixel reconstruction. Subsequently, positions, embeddings, and the frozen encoder are transferred onto the drone.

Online, the drone's view is captured through a camera positioned beneath it. The outlines of the RGB image are then extracted and forwarded to the encoder, which generates an embedding. This embedding is subsequently compared, using cosine similarity, to all embeddings derived from the reference images. The drone's position is then inferred based on the position of the reference image with the highest similarity score.

<img src="https://github.com/theodpzz/uav-localization/blob/main/figures/overview_method_final.png" alt="Method overview" width="600">

---

## Getting Started

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/uav-localization.git
```

### Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

### Training

To train the model, run the following command:

```bash
python train.py
```

### Data

Example data is available in example-data folder.

> [!WARNING]  
> These are toy examples and do not correspond to the dataset used.

train.csv, test.csv contain filenames and coordinates of each tile.
Below the structure of the data folder for n reference tiles and m uav views.

```bash
├── train/
│   ├── reference_tile_1.npy
│   ├── reference_tile_2.npy
│   ├── ...
│   └── reference_tile_n.npy
│
├── test/
│   ├── uav_view_1.npy
│   ├── uav_view_2.npy
│   ├── ...
│   └── uav_view_m.npy
│
├── train.csv
└── test.csv
```

---

## Citation

> [!IMPORTANT]  
> This project is based on the work by Di Piazza et al. If you use this code in your research, we would appreciate the following citation:

```BibTeX
@inproceedings{dipiazza2024uavloc,
  author    = {Di Piazza Theo, Meinhardt-Llopis Enric, Facciolo Gabriele, Bascle Benedicte, Abgrall Corentin and Devaux Jean-Clement},
  title     = {Leveraging neural networks and edge detection for better UAV localization},
  booktitle = {Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2024},
  organization = {IEEE},
}
```
