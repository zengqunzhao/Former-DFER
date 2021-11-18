# Former-DFER

*Zengqun Zhao, Qingshan Liu. "[Former-DFER: Dynamic Facial Expression Recognition Transformer](https://zengqunzhao.github.io/doc/pdfs/MM2021.pdf)". ACM International Conference on Multimedia.*

## Requirements

- pytorch==1.8.0
- torchvision==0.9.0

## Training

- Step 1: download [DFEW](https://dfew-dataset.github.io) dataset.
- Step 2: fill in the ***your_DFEW_Face_path*** in ```script.py```, then run ```script.py```.
- Step 3: run ``` sh DFEW_Five_Fold.sh ```

## Pre-trained Models

The pre-trained Former-DFER model on DFEW can be downloaded [here](https://drive.google.com/file/d/1YV-KpdYQVAvSQw1setzBF1LeT4qx1bVt/view?usp=sharing).

## Citation

```
@inproceedings{zhao2021former,
  title={Former-DFER: Dynamic Facial Expression Recognition Transformer},
  author={Zhao, Zengqun and Liu, Qingshan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1553--1561},
  year={2021}
}
```
