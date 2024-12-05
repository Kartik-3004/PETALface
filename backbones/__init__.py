from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        return iresnet18(lora_rank=r, lora_scale=scale, pretrained=False, progress=True, use_lora=use_lora, **kwargs)
    elif name == "r34":
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        return iresnet34(lora_rank=r, lora_scale=scale, pretrained=False, progress=True, use_lora=use_lora, **kwargs)
    elif name == "r50":
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        return iresnet50(lora_rank=r, lora_scale=scale, pretrained=False, progress=True, use_lora=use_lora, **kwargs)
    elif name == "r100":
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        return iresnet100(lora_rank=r, lora_scale=scale, pretrained=False, progress=True, use_lora=use_lora, **kwargs)
    elif name == "r200":
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        return iresnet200(lora_rank=r, lora_scale=scale, pretrained=False, progress=True, use_lora=use_lora, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        from .vit import VisionTransformer
        return VisionTransformer(
            lora_rank=r, lora_scale=scale, img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24, 
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True, use_lora=use_lora)

    elif name == "vit_b_iqa":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        from .vit_iqa import VisionTransformer
        return VisionTransformer(
            lora_rank=r, lora_scale=scale, img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24, 
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True, use_lora=use_lora)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
        
    elif name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True)

    elif name=="swin_256new":
        num_features = kwargs.get("num_features", 512)
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        from .swin_models import  SwinTransformer
        kwargs['reso']=120
        return  SwinTransformer(lora_rank=r, lora_scale=scale, img_size=120, patch_size=6, in_chans=3, num_classes=512,
                 embed_dim=384, depths=[2,18,2], num_heads=[ 8, 16,16],
                 window_size=5, use_lora=use_lora, **kwargs) 

    elif name=="swin_256new_iqa":
        num_features = kwargs.get("num_features", 512)
        r = kwargs.pop("r", 4)
        scale = kwargs.pop("scale", 1)
        use_lora = kwargs.pop('use_lora', False)
        from .swin_models_iqa import  SwinTransformer
        kwargs['reso']=120
        return  SwinTransformer(lora_rank=r, lora_scale=scale, img_size=120, patch_size=6, in_chans=3, num_classes=512,
                 embed_dim=384, depths=[2,18,2], num_heads=[ 8, 16,16],
                 window_size=5, use_lora=use_lora, **kwargs) 


    else:
        raise ValueError()
