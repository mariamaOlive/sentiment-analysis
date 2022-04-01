from unittest.util import _MAX_LENGTH
from transformers import BertTokenizer
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# can be up to 512 for BERT
MAX_LENGTH = 256
BATCH_SIZE = 1
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


def plot_acuracy_loss(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = MAX_LENGTH, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def encode_examples(texts, labels, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    # for review, label in tfds.as_numpy(ds):
    for text, label in zip(texts, labels):
        bert_input = convert_example_to_feature(text)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


def get_bert_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # train dataset
    ds_train = encode_examples(X_train, y_train).shuffle(3).batch(BATCH_SIZE)

    # test dataset
    ds_test = encode_examples(X_test, y_test).batch(BATCH_SIZE)

    #validation dataset
    ds_valid = encode_examples(X_valid, y_valid).batch(BATCH_SIZE)

    return ds_train, ds_valid, ds_test
    

def get_test_metrics(model, ds_test, y_test):

    #Predictin test dataset
    tf_output = model.predict(ds_test)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    # labels = ['Negative','Positive'] #(0:negative, 1:positive)
    label = tf.argmax(tf_prediction, axis=1)
    label_pred = label.numpy()
    # print(label_pred)

    print(classification_report(y_test, label_pred))

    print(confusion_matrix(y_test, label_pred))

    return label_pred