import tensorflow as tf
import numpy as np
import pandas as pd

from keras import models
from keras.models import load_model

#from keras import layers

checkpoint_path = "training_1/cp.ckpt"
model = create_model()

model.load_weights(checkpoint_path)

model = load_model('write the name of the model')

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
              metrics = ['accuracy'])


# history = model.fit(x_train,y_train,epochs = 50 , validation_data = (x_val, y_val))

# predictions = model.predict_classes(x_val)
# predictions = predictions.reshape(1,-1)[0]
# print(classification_report(y_val, predictions, target_names = labels))
from keras.preprocessing import image

test_image = image.load_img(imagePath, target_size = (224,224)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)