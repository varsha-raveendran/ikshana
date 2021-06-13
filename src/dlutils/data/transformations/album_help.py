import inspect
import operator
import albumentations as A
class HELPER:
    
    album_list = ['Blur', 'CoarseDropout', 'ColorJitter', 'Cutout', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'GaussianBlur',  'Normalize']

    def __init__(self):
        for album in self.album_list:
            setattr(self, album, self.func_details(album))

    def functions(self, name=None):
        if name == None:
            print('use <object_name>.<function_name> to know more about each. Eg: <object_name>.Blur')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("1. 'Blur' - Blur the Input Image using Rnadom Size Kernel.")
            print("2. 'CoarseDropout' - CoarseDropout of Rectangular Regions in the Image.")
            print("3. 'ColorJitter' - Randomly changes the brightness, contrast, and saturation of an Image.")
            print("4. 'Cutout' - CoarseDropout of the square regions in the Image.")
            print("5. 'Flip' - Flip the Image either horizontally, vertically or both horizontally and vertically.")
            print("6. 'HorizontalFlip' - Flip the Image horizontally around the y-axis")
            print("7. 'VerticalFlip' - Flip the Image vertically around the x-axis")
            print("8. 'GaussianBlur' - Blur the input Image using a Gaussian filter with a random kernel size.")
            print("9. 'Normalize' - Normalization is applied by the formula: **img = (img - mean * max_pixel_value) / (std * max_pixel_value)**.")
            print("Note: To Use Extra Albumentations Directly send the Albumentation Function Object to Dataser Trasnform.")

    def func_details(self, name, docs=False):
        func = operator.attrgetter(name)
        params = {p[1].name: p[1].default for p in inspect.signature(func(A)).parameters.items()}
        detail = f" {{{name}: {params}, ... }}"
        if docs:
            detail += '\n', func(A).__doc__
        
        return detail