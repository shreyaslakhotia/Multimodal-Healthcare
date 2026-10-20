import numpy as np
from PIL import Image
from icu_mm.data.cxr import preprocess_image


def test_preprocessing_geometry_and_xrv_range_without_fitting():
    for level,expected in [(0,-1024),(255,1024)]:
        image=Image.fromarray(np.full((100,200),level,dtype=np.uint8))
        tensor,pixels=preprocess_image(image)
        assert tensor.shape==(1,224,224) and pixels.shape==(224,224)
        assert (tensor.numpy()==expected).all()
