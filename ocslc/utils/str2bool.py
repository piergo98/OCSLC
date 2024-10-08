import argparse

def str2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
    raise argparse.ArgumentTypeError('Boolean value expected.')