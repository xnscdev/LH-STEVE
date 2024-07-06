from mineclip import MineCLIP


def load_mineclip(ckpt_path, device):
    model = MineCLIP(arch='vit_base_p16_fz.v2.t2', hidden_dim=512, image_feature_dim=512, mlp_adapter_spec='v0-2.t0', pool_type='attn.d2.nh8.glusw', resolution=(160, 256)).to(device)
    model.load_ckpt(ckpt_path, strict=True)
    model.eval()
    return model
