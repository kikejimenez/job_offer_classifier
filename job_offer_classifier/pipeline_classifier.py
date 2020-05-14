# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_pipeline_classifier.ipynb (unless otherwise specified).

__all__ = ['split_dataframe', 'balanced_labels_in_split', 'tf_input_fn', 'train_input_fs', 'predict_input_fs',
           'embedded_text_feature_column_f', 'load_estimator', 'train', 'f1_score', 'evaluate', 'predict', 'sentiment',
           'plot_confusion_matrix', 'export_estimator', 'Pipeline']

# Cell
import shutil
from random import randint
from os import path,listdir
from shutil import rmtree
from tempfile import TemporaryDirectory
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import tensorflow_hub as hub
import tensorflow as tf

# Cell


def split_dataframe(df, **kwargs):
    '''Split with the pandas `sample` method.
    '''
    train = df.sample(**kwargs)
    test = df.drop(train.index)
    return {'train': train, 'test': test}



def balanced_labels_in_split(df, **kwargs):
    '''Give a balanced train-test split in 'positive' and 'negative' labels
    '''
    positives = split_dataframe(df[df.sentiment == 1], **kwargs)
    negatives = split_dataframe(df[df.sentiment == 0], **kwargs)
    train = pd.concat([positives['train'], negatives['train']])
    test = pd.concat([positives['test'], negatives['test']])

    return {'train': train, 'test': test}

# Cell
def tf_input_fn(df, **kwargs):
    ''' Load a TF function for a DataFrame
    '''
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        df, df["sentiment"],  **kwargs
    )

def train_input_fs(**kwargs):
    '''TF functions with parameters for training
    '''
    return {
        name: tf_input_fn(df, shuffle=True, num_epochs=None)
        for name, df in kwargs.items()
    }

def predict_input_fs(**kwargs):
    '''TF with parameters for testing
    '''
    return {
        name: tf_input_fn(df,shuffle=False)  for name, df in kwargs.items()
    }

# Cell
def embedded_text_feature_column_f(module_spec="https://tfhub.dev/google/nnlm-en-dim128/1"):
    ''' Call the text embedding from the TF-Hub library
    '''
    return hub.text_embedding_column(
        key="payload", module_spec=module_spec
    )

# Cell
def load_estimator(
    embedded_text_feature_column, estimator_f=tf.estimator.DNNClassifier,model_dir = None
):
    ''' Load the TF `estimator`
    '''
    return estimator_f(
        model_dir = model_dir,
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.003)
    )

# Cell
def train(estimator, train_input_fn,steps=5000):
    '''Train with TF `estimator.train`
    '''
    result = {}
    for name,input_fn in train_input_fn.items():
        result[name] = estimator.train(input_fn=input_fn, steps=steps)
    return result

# Cell
def f1_score(estimations):
    ''' Calculates function: \n
    *f1_score(precision,recall) =  (2 x precision x recall) / (precision + recall)*
    '''
    precision =  estimations['precision']
    recall = estimations['recall']
    if (precision + recall) < 10**(-12):
        return 0.0

    return 2.0 * precision * recall / (precision + recall)

# Cell
def evaluate(estimator, **args:tf_input_fn):
    '''Evaluate with TF `estimator.evaluate`
    '''
    results = {}
    for name,input_fn in args.items():
        results[name] = estimator.evaluate(input_fn=input_fn)
        results[name]['f1_score'] = f1_score(results[name])
    return results

# Cell
def predict(estimator, df_examples):
    ''' Predict with TF `estimator.predict` and from a dataframe of payloads.
    '''

    def predict_from_input_fn(estimator, **input_fns):
        res = {}
        for name, input_fn in input_fns.items():
            res[name] = np.array(
                [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]
            )
        return res

    return predict_from_input_fn(
        estimator, **predict_input_fs(examples=df_examples)
    )['examples']

# Cell
def sentiment(estimator, doc):
    ''' Gets the sentiment of the *doc* string
    '''
    ex_df = pd.DataFrame([{'payload':doc,'sentiment':0}])
    pred = ["negative", "positive"][predict(estimator,ex_df)[0]]
    return pred #doc + '\n --> \n' + pred

# Cell
def plot_confusion_matrix(df_data, estimator, input_fn, header):
    ''' Plot the Confusion Matrix: ${{TN,FN},{FP,TP}}$
    '''
    def get_predictions(estimator, input_fn):
        return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

    LABELS = ["negative", "positive"]

    # Create a confusion matrix on dataframe data.
    cm = tf.math.confusion_matrix(
        df_data["sentiment"], get_predictions(estimator, input_fn)
    )

    # Normalize the confusion matrix so that each row sums to 1.
    cm = tf.cast(cm, dtype=tf.float32)
    cm = cm / tf.math.reduce_sum(cm, axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.title(header)
    plt.xlabel("Predicted")
    plt.ylabel("True")

# Cell
def export_estimator(estimator, dst_estimator):
    ''' Copy the estimator's directory to a new directory
    '''
    shutil.copytree(estimator.model_dir, dst_estimator)

# Cell
class Pipeline:
    '''Implements the workflow: Load -> Train -> Evaluate.'''
    def __init__(
        self,
        src_file=None,
        estimator_dir=None,
        frac=0.60,
        train_steps=5000,
        random_state=None
    ):
        '''Pass the arguments to class attributes.
           Load and split the data.
        '''
        self.frac = frac
        self.estimator_dir = estimator_dir
        self._is_estimator_dir = path.isdir(str(estimator_dir))
        self.random_state = random_state
        self.train_steps = train_steps
        self.module_spec = "https://tfhub.dev/google/nnlm-en-dim128/1"

        if src_file is not None:
            self.data = pd.read_csv(src_file)
            self.data['sentiment'] = self.data.sentiment.apply(
                lambda x: int(x == 'positive')
            )
            self.split_dataset()

    def __del__(self):
        ''' Removes the `estimator` and its corresponding directory,
        unless the estimator_dir is None.
        '''
        if not self._is_estimator_dir:
            rmtree(self.estimator_dir, ignore_errors=True)

    def split_dataset(self):
        ''' Train-test splits. Deletes empty dataframes.
        '''
        self.dfs = balanced_labels_in_split(
            self.data, random_state=self.random_state, frac=self.frac
        )
        if self.dfs['test'].shape[0] == 0:
            del self.dfs['test']
        if self.dfs['train'].shape[0] == 0:
            del self.dfs['train']

    def input_fns(self):
        self.input = {}
        if 'train' in self.dfs.keys():
            self.input['train'] = train_input_fs(train=self.dfs['train'])

        self.input['predict'] = predict_input_fs(**self.dfs)

    def load_estimator(self):
        self.embedded_text_feature_column = embedded_text_feature_column_f(
            module_spec=self.module_spec
        )
        self.estimator = load_estimator(
            self.embedded_text_feature_column, model_dir=self.estimator_dir
        )
        self.estimator_dir = self.estimator.model_dir

    def train(self):
        if self.input['train'] is not None:
            train(self.estimator, self.input['train'], steps=self.train_steps)

    def evaluate(self):
        self.evaluation = evaluate(self.estimator, **self.input['predict'])

    def plot_confusion_matrix(self, label):
        plot_confusion_matrix(
            self.dfs[label],
            self.estimator,
            self.input['predict'][label],
            header=label + ' data'
        )

    def export_estimator(self, dst_dir):
        try:
            _ = self.estimator
        except:
            self.load_estimator()
        export_estimator(self.estimator, dst_dir)

    def predict(self, df_examples):
        '''Predict from dataframe'''
        return predict(self.estimator, df_examples)

    def sentiment(self, doc):
        return sentiment(self.estimator, doc)

    def pipeline(self):
        ''' The pipeline flow is:
            input_fns --> load_estimator --> train --> evaluate
        '''
        self.input_fns()
        self.load_estimator()
        self.train()
        self.evaluate()