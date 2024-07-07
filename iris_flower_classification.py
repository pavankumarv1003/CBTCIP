import tensorflow as tf
import pandas as pd
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

features_col= ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
lables= ['Setosa','Versicolor','Virginica']

url= "https://docs.google.com/spreadsheets/d/e/2PACX-1vQKk1WzH_R4rP3GKjzk2dpZoaC5DG8w5spb_dO4KVtbp_06tDqBZBR4hsAg-uExExJNq2XMDX3vKrFY/pub?gid=1772486447&single=true&output=csv"
url_data= pd.read_csv(url, names=features_col, header=0)

url_lables=url_data.pop('Species')
train,test,train_lables,test_lables=train_test_split(url_data,url_lables,test_size=0.2)


map={'Iris-setosa':0 ,'Iris-versicolor':1,'Iris-virginica':2}
train_lables=train_lables.map(map)
test_lables=test_lables.map(map)

def input_func(featurs,labels,training=True,batch_size=256):
  dataset=tf.data.Dataset.from_tensor_slices((dict(featurs),labels))

  if training==True:
    dataset=dataset.shuffle(10000).repeat()
  return dataset.batch(batch_size)

fc=[]
for k in train.keys():
  fc.append(tf.feature_column.numeric_column(key=k))

model=tf.estimator.DNNClassifier(feature_columns=fc, hidden_units=[30,10],n_classes=3)

model.train(input_fn=lambda: input_func(train, train_lables, training=True),steps=20000)

eval_result = model.evaluate(input_fn=lambda: input_func(test, test_lables, training=False))
clear_output()
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    while True:
        val = input(feature + ": ")
        try:
            predict[feature] = [float(val)]
            break
        except ValueError:
            print("Please enter a valid number.")

predictions = model.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print()
    print(lables[class_id])
    print("prediction percentage:",probability*100)

