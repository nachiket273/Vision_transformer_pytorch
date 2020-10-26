# Vision_transformer_pytorch
Pytorch Implementation of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>

<img src="/img/vit.png" width="500px"></img>

Image patches are generated as:

```python
b, c, h, w = img.shape
img = img.permute(0, 2, 3, 1)
patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
patches = patches.reshape(b, -1, c * patch_size * patch_size)
```

#### Image:
<img src="/img/img.png" width="500px"></img>

#### Patches:
<img src="/img/patches.png" width="500px"></img>


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
