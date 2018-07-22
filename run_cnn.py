import numpy as np
import tensorflow as tf
import h5py as h5
from cnn import GTSRB_Classifier

hf = h5.File('./data/gtsrb_data.h5', 'r')
data = {k: np.array(v) for k,v in hf.items()}
hf.close()

gtsrb_cl = GTSRB_Classifier()

gtsrb_cl.create_classifier(model_dir='./models')

gtsrb_cl.train(train_data=data['train_data'].astype(np.float32), train_labels=data['train_labels'].astype(np.int32), num_steps=20000, batch_size=300)
