import numpy as np
from copy import deepcopy
from typing import Union
from collections.abc import Iterable
import numpy as np
import os

def flatten(dictionary, level=[]):
    """
    Copied from Andrew Sedler's lfads_torch.utils
    --------------------------------

    Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def indicator_func_to_matrix(size,indicator_func,dtype=bool):
    A = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            A[i,j] = indicator_func(i,j)
    return A.astype(dtype)

def make_path_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def duplicate_list(item,repeats):
    return [item] * repeats

def set_attributes(object, **kwargs):
    if not len(kwargs):
        return object
    for attr,val in kwargs.items(): #zip(attributes,values):
        setattr(object,attr,val)
    return object

def copy_attributes(from_object,to_object,attributes):
    for attr in attributes:
        setattr(
            to_object,
            attr,
            getattr(from_object,attr)
        )
    return to_object

def empty_call(func):
    return func()

def select(obj,**kwargs):
    return obj.select(**kwargs)


#
# class DataArrayCoordsWithList(xr.DataArray):
#     def __init__(self):


setattrs = lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)]
setattrs_kwargs = lambda target,**kwargs: setattrs(target,*list(zip(*kwargs.items())))



batch_choice = lambda p:  np.stack([np.random.choice(p.shape[1],p=p[i]) for i in range(len(p))])

normalise = lambda array,axis=None: array/array.sum(axis=axis,keepdims=True)


if __name__ == '__main__':
    # class A:
    #     def __init__(self):
    #         return
    # M = A()
    # set_attributes(A,**{'a':1,'b':2})

    # a = HMM_Dataset(np.ones((3,2,2)))
    # print(a.split([1],axis=2)[0]())
    # print()
    # A = HMMData(

    A = xr.DataArray(
        np.random.uniform(size=(5,10,20)),
        dims=('trials','time','neurons'),
        coords={
            'trials' : ['train']*2+['test']*3,
            'neurons': ['heldin']*10+['heldout']*10

            # 'neuron_part':['heldin']*10+['heldout']*10,
        }
    )


    # A.coords['heldin'] = ('neurons',np.arange(20) < 10)
    # A.coords['heldout'] = ('neurons', np.arange(20) > 10)
    # print(A.sel(neurons='heldin'))
    # print(A.loc['train',:,'heldin'].shape)
    # print(A.sel(trials='train',neurons_part=['heldin','heldout']).shape)
    # print(A.groupby('neurons')['heldout'].shape)
    # print(A.sel(heldin=True))
    # print(A.to_dataframe(name='a'))
    # print(A.loc[:,:,'heldin':'heldout'].shape)
    # print(A())

    A = xr.DataArray(
        np.random.uniform(size=(5, 10, 20)),
        dims=('trials', 'time', 'neurons'),
        # coords={
        #     'trials': np.arange(5),
        #     'time'  : np.arange(10),
        #     'neurons': np.arange(20)
        #     # 'neuron_part':['heldin']*10+['heldout']*10,
        # }
    )

    B = xr.Dataset(
        {
            'data': A,
            'trial_type': xr.DataArray(['train'] * 2 + ['test']*3,dims='trials'),
            'neuron_type': xr.DataArray(['heldin'] * 10 + ['heldout'] * 10, dims='neurons')
        }
    )
    # print(B.where(B.trial_type == 'train').data.shape)
    # print(B.where(B.neuron_type in ['heldin','heldout']).data.shape)

    data = xr.DataArray(
        np.arange(15),
        coords={
            'trials': ['train'] * 5 + ['val'] * 5 + ['test'] * 5
        }
    )
    # print(data.sel(trials='train'))              # works
    # print(data.sel(trials=['train','test']))     # doesnt work

    data = xr.DataArray(
        np.arange(15),
        dims = ('trials'),
        coords={
            # 'trials': np.arange(15),
            'trial_split':('trials',['train'] * 5 + ['val'] * 5 + ['test'] * 5),
        }
    )
    mask1 = data['trial_split'].isin(['train','test'])
    mask2 = data.trial_split.isin(['train','test'])
    # mask =
    # print(data[mask])
    # print(mask1 & True )

    data = MultiSelectDataArray(
        np.ones((10,20,30)),
        dims = ('trials','time','neurons'),
        coords={
            # 'trials': np.arange(15),
            'trial_split':('trials',['train'] * 5 + ['val'] * 2 + ['test'] * 3),
            'neurons_split':('neurons',['heldin']*10+['heldout']*20)
        }
    )

    print(data.select(trial_split=['train','test'],neurons_split='heldin').shape) #

    data = xr.DataArray(
        np.ones((15,20,30)),
        dims = ('trials','time','neurons'),
        coords={
            'train': ('trials',np.arange(15) <  5),
            'val':   ('trials',(np.arange(15) >= 5)&(np.arange(15) < 10)),
            'test':  ('trials', np.arange(15) >= 10),
            'heldin':('neurons',np.arange(30)<15)
        }
    )
    print(data.sel(trials=data['train'],neurons=data['heldin']).shape)
    # print(data.where(data['train']).shape)
    # print(data.trial_split.dims[0])
    # b = {'a':1}
    # print(b.popitem())
    # print(dir(b))
    print(np.reshape(data.values,-1))
