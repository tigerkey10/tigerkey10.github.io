#경고 무시 
import warnings
warnings.simplefilter('ignore')

#자주 사용하는 패키지를 임포트 
import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd 
import statsmodels.api as sm
import sklearn as sk

#맷플롯립 설정
mpl.use('Agg')

#시본 설정
sns.set()
sns.set_style('whitegrid')
sns.set_color_codes()