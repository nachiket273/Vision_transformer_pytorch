# Vision_transformer_pytorch
Pytorch Implementation of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>

<img src="/img/vit.png" width="500px"></img>

## Usage

```python
import torch
from vit import ViT

model = ViT(
    img_size=224,
    patch_size=16,
    in_ch=3,
    num_classes=1000,
    use_mlp=True,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_rate=0.0,
)

img = torch.randn(1, 3, 224, 224)
pred = model(img)
```

## Pretrained Weights

* Used Imagenet-1k pretrained weights from https://github.com/rwightman/pytorch-image-models/
* Updated checkpoint for this implementation and new weights can be found on <a href="https://drive.google.com/file/d/1xY9gk_KUoXfJkNiJ3L7W9xYWpngp4FHY/view?usp=sharing">drive</a> location.

## Citations

```bibtex
@inproceedings{
    anonymous2021an,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=YicbFdNTTy},
    note={under review}
}
