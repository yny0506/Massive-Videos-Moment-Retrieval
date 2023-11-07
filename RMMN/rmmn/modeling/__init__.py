from .rmmn import MMN
ARCHITECTURES = {"MMN": MMN}

def build_model(cfg):
    if cfg.MODEL.ARCHITECTURE == 'MMN':
        return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
    
