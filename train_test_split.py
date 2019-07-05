import os

import numpy as np
import pandas as pd

from params import *

# reading data
df = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))

msk = np.random.rand(len(df)) < 0.7
train_df = df[msk]
test_df = df[~msk]

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)