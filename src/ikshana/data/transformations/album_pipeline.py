# data/transformations/album_pipeline.py

import operator
from typing import OrderedDict, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

def generate_compose(mean: Union[tuple,list]= None, std: Union[tuple,list]= None, **kwargs: OrderedDict) -> A.Compose:
    '''
    To generate a Albumentation Compose method, by creating list of all user provided trawsnformations.
    To Know about trasnformations, please user transformations.album_helper.HELPER class.

    Args:
        mean: The tuple/list of Mean Values of all the Image Channels
        std: The tuple/list of Standard Deviation Values of all the Image Channels
        **kwargs: Key Word Arguments.
            eg: {'Blur' : {'blur_limit': 7, 'always_apply': False, 'p': 0.5}}
            For Multiple Transformations
            eg: {
                'CoarseDropout': {'max_holes': 8, 'max_height': 8, 'max_width': 8, 'p': 0.5},
                'Blur' : {'blur_limit': 7, 'always_apply': False, 'p': 0.5}
                }
    Return:
        Albumentation.Compose Function with List of Transformation Provided + ToTensor.
    '''
    trans = []

    # Adding Each Trasnformation.
    for k,v in kwargs.items():
        
        key = k
        value = v
        if k == 'Sequential':
            seq_trans = []
            p = 0.5
            for i,j in v.items():
                if i == 'p':
                    p = j
                    continue
                seq_trans.append(operator.methodcaller(i, **j)(A))
            key = 'Sequential'
            value = {'transforms': seq_trans, 'p': p}

        trans.append(operator.methodcaller(key, **value)(A))

    # For Normalizing if not given as Part of kwargs.
    if 'Normalize' not in kwargs and mean != None and std != None:
        trans.append(A.Normalize(mean, std))
    trans.append(ToTensorV2())
    
    return A.Compose(trans)


# Testing
if __name__ == '__main__':
    q = {
        'Sequential': {'PadIfNeeded': {
                            'min_height': 40,
                            'min_width': 40,
                            'border_mode': cv.BORDER_CONSTANT,
                            'value': (0.4914, 0.4822, 0.4465)
                        },
                        'RandomCrop': {
                            'height': 32,
                            'width': 32
                        },
                        'p': 1.0
                    },
        'CoarseDropout': {'max_holes': 8, 'max_height': 8, 'max_width': 8, 'p': 0.5},
        'Blur' : {'blur_limit': 7, 'always_apply': False, 'p': 0.5}
    }
    
    print(generate_compose([1,1,1], [1,1,1], **q))