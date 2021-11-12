import fertility_classification
import numpy as np
import os


tree = fertility_classification.create_tree()
datapoint = np.array(fertility_classification.gather_user_data()).reshape(1, -1)
#datapoint = np.array([-0.33,0.94,1,0,1,0.8,1,0.31]).reshape(1, -1)
diagnosis = tree.predict(datapoint)[0]
os.system('cls')
if diagnosis == 0:
    print("You are fertile!")
else:
    print("You are infertile!")
