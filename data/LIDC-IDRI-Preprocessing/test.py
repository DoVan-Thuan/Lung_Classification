import numpy as np
import pandas as pd

a = pd.read_csv('/Users/thanhdo/Projects/LungCancer/Lung_Classification/data/LIDC-IDRI-Preprocessing/data/Meta/meta.csv')
img = "0001_NI000_slice000"
label = a[a['original_image']==img]['malignancy'][0]
print(label)