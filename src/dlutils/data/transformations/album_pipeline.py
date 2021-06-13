import operator
from typing import OrderedDict, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        trans.append(operator.methodcaller(k, **v)(A))

    # For Normalizing if not given as Part of kwargs.
    if 'Normalize' not in kwargs and mean and std:
        trans.append(A.Normalize(mean, std))
    trans.append(ToTensorV2())
    
    return A.Compose(trans)
