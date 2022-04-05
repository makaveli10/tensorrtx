"""
Converts pytorch weights file to tensorrt wts file.
"""

import argparse
import struct

from timm.models import create_model


parser = argparse.ArgumentParser(description='Convert torch weights to tensorrt wts.')
parser.add_argument('model_name', help='timm model name')
parser.add_argument('--wts', help='path to output file')

def convert_torch_weigths_to_wts(args):
    """
    """
    # Load model
    model = create_model(
        args.model_name,
        num_classes=1000,
        in_chans=3,
        pretrained=True)
    model.float().to('cpu').eval()

    with open(args.wts, "w") as fi:
        fi.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            fi.write(f'{k} {len(vr)} ')
            for vv in vr:
                fi.write(' ')
                fi.write(struct.pack('>f',float(vv)).hex())
            fi.write('\n')

if __name__=="__main__":
    args = parser.parse_args()
    convert_torch_weigths_to_wts(args)
