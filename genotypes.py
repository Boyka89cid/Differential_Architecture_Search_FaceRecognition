'''
Below Code has been referred from https://github.com/quark0/darts
'''
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTSCIFAR = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTSCELEB_SM = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
DARTS_LFW = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_7x7', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_7x7', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))