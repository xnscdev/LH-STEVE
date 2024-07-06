import lightning as L
from mineclip import MineCLIP


class MineCLIPWrapper(L.LightningModule):
    def __init__(self, ckpt_path):
        super().__init__()
        self.mineclip = MineCLIP(arch='vit_base_p16_fz.v2.t2', hidden_dim=512, image_feature_dim=512, mlp_adapter_spec='v0-2.t0', pool_type='attn.d2.nh8.glusw', resolution=(160, 256))
        self.mineclip.load_ckpt(ckpt_path, strict=True)

    def forward(self, x):
        return self.mineclip.encode_video(x)
