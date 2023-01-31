# Dataset class에서 전처리를 수행해주는 함수들을 정의한다.
import numpy as np

#-- Compose -> transform 함수들 여러개 묶어서 진행
class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        
        for t in self.transforms:
            img = t(img)
        return img


#---------
# numpy ndarray transform
#---------
class Normalize:
    ''' 정규화 '''
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshpae(*mshape)

        if not np.isscalar(std):
            reshape = [1] * array.ndim
            reshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*reshape)
        return (array - mean) / std

class Flatten:
    ''' 펼치기 '''
    def __call__(self, array):
        return array.flatten()

class AsType:
    ''' type change '''
    def __init__(self, dtype=np.float32):
        self.dtype=dtype

    def __call__(self, array):
        return array.astype(self.dtype)

ToFloat = AsType

class Toint(AsType):
    def __init__(self, dtype=np.int):
        self.dtype = dtype
