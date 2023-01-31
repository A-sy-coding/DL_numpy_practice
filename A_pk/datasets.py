import numpy as np
import os

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data

    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)
    
    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

#-- Dataset class -> 전처리 기능도 확장
class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # transform이 none이면 아무런 전처리를 하지 않고 그대로 반환한다.
        if self.transform is None:
            self.transform = lambda x : x
        if self.target_transform is None:
            self.target_transform = lambda x : x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index) # index는 정수만 지원하도록 한다.
        if self.label is None:
            return self.transform(self.data[index]), None

        else:
            return self.transform(self.data[index]), \
                    self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

#-- Bigdata일 때 데이터 처리
class BigData(Dataset):
    '''
    초기화할 때에는 데이터를 읽지 않고,
    데이터에 접근할 때 데이터를 읽도록 하여, 메모리 절약을 할 수 있다.
    '''
    def __getitem__(self, index):
        x = np.load('data/{}'.format(index))
        t = np.load('label/{}'.format(index))
        return x, t

    def __len__(self):
        return len(self.data)

#-- Spiral dataset settings
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
