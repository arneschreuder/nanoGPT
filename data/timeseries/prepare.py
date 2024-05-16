import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

TRAIN_VAL_SPLIT = 0.9
CYCLES = 10
FEATURES = 1

data = []

for i in range(CYCLES * 360):
    rad = math.radians(i)
    x = rad
    y = (math.sin(x) + 1) / 2
    data += [y]

print(data[:10])

# Visualise data
plt.plot(data)
plt.savefig(os.path.join(os.path.dirname(__file__), "data.png"))

# create the train and test splits
N = len(data)
train_data = data[: int(N * TRAIN_VAL_SPLIT)]
val_data = data[int(N * TRAIN_VAL_SPLIT) :]

# export to bin files
train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)

train_data.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_data.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {"vocab_size": FEATURES, "train_val_split": TRAIN_VAL_SPLIT, "cycles": CYCLES}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
