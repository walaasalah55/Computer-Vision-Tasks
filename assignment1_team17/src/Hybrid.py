from utils import *
from src.filters import *
from src.EdgeDetection import *


def hybrid_image(img1,img2):

    hpass_image=highpass_filter(img1)
    lpass_image=lowpass_filter(img2)
    hybrid=np.add(hpass_image,lpass_image)

    return hybrid



