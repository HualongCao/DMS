# DMS: Low-overlap Registration of 3D Point Clouds with Double-layer Multi-scale Star-graph

[DMS: Low-overlap Registration of 3D Point Clouds with Double-layer Multi-scale Star-graph](https://ieeexplore.ieee.org/document/10530423).

[Hualong Cao](https://scholar.google.com/citations?user=Vh-QasEAAAAJ), [Yongcai Wang](https://in.ruc.edu.cn/yjdw_yc/js/ycw.html), and Deying Li.

## Pipline
![](assets/pipline.png)

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n dms python==3.8
conda activate dms
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```


## Citation

```bibtex
@ARTICLE{DMS,
  author={Cao, Hualong and Wang, Yongcai and Li, Deying},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={DMS: Low-overlap Registration of 3D Point Clouds with Double-layer Multi-scale Star-graph}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TVCG.2024.3400822}
}

```

## Acknowledgements

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [Maximal-Cliques](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
