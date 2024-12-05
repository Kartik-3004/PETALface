from .partial_fc import PartialFC_V2

def get_head(name, **kwargs):
    if name == "partial_fc":
        return PartialFC_V2(**kwargs)
    else:
        raise ValueError("Head not Implemented")