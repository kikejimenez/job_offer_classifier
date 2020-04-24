# Sentiment Classifier
> Classification of email job offer response emails 


This project classifies job offer response emails as 'positive' or 'negative' according to whether an email response expresses an interest in a job offer. The dataset contains job offer response emails annotated with 'positive' and 'negative' labels. The positive labels represent an interest in a job offer.

## Install

### Pip

The sentiment classifier can be found on PyPI so you can just run:

```shell
pip install job-offer-classifier
```

### Github

Clone the [GitHub](https://github.com/kikejimenez/job_offer_classifier) repository and `cd` to the cloned repo directory. Install the depencies in the `requeriments.txt`. The use of *REPL* or  *Notebook* is strongly  recommended

## How to use

### Run the Pipeline

First load and run the data science pipeline by importing the module:

```python
from job_offer_classifier.pipeline_classifier import Pipeline
```

Instantiate the class `Pipeline` and call the `pipeline` method. This method loads the dataset, and trains and evaluates the model. The source file is the annotated dataset of payloads.

```python
pl = Pipeline(src_file = '../data/interim/payloads.csv',random_state=931696214)
pl.pipeline()
```

The parameter `random_state` is the pandas seed used in the dataframe split. This parameter is necessary to present deterministic results and has been chosen from the results of the [k fold validation](#K-fold-validation).

### Predict Job Offer Sentiments

To make a prediction, use the `sentiment` method

```python
pl.sentiment(''' Thank you for offering me the position of Merchandiser with Thomas Ltd.
I am thankful to accept this job offer and look ahead to starting my career with your company
on June 27, 2000.''')
```




    'positive'



One can take an example from the test set, contained in the `dfs` attribute. This attribute is a dictionary of  pandas dataframes.

```python
example = pl.dfs['test'].sample(random_state=1213702178).payload.iloc[0]
print(example.strip())
```

    thank you for offering me the position of financial analyst at Lozano-Carlson.
    i was delighted to meet
    you and learn more about the company.
    although i verbally agreed to accept the position, i have given it a lot of thought and decided to turn
    down the post.
    i believe it is in my, and your company’s, best interests.
    ultimately, i elected to take on a
    position at a firm where i believe my skills and experience are a better fit. i truly apologise for any
    inconvenience i have caused.
    i was impressed with Lozano-Carlson during the interview, and continue to be at this time.
    wishing you
    all the best in the future and hope to still see you in attendance at the snow terrace financial conference
    in june.


```python
pl.sentiment(example)
```




    'negative'



## Performance

We use two tools to assesss the performance of the model:
  - Confusion Matrix 
  - K fold Validation

### Confusion matrix

To plot the confusion matrix, the `Pipeline` has the method `plot_confusion_matrix`.

```python
pl.plot_confusion_matrix('train')
```


![png](docs/images/output_24_0.png)


```python
pl.plot_confusion_matrix('test')
```


![png](docs/images/output_25_0.png)


The percentage of the cases that are negative and predicted positive (*False Negative rate*) tend to be greater than the percent of the cases that are positive and predicted negative (*True Negative rate*).  This is consistent with that fact that the dataset has more positive than negative cases and the model tends to see more positives.


### K fold validation

To assess the performance of the model via the k fold validation method, import the class [`KFoldPipe`](/job_offer_classifier/validations#KFoldPipe)

```python
from job_offer_classifier.validations import KFoldPipe
```

Run the `k_fold_validation` method

```python
kfp = KFoldPipe(src_file='../data/interim/payloads.csv',n_splits=4)
kfp.k_fold_validation()
```

The averaged scores are stored in `averages`

```python
kfp.averages['train']
```




    {'accuracy': 0.9926739931106567,
     'accuracy_baseline': 0.7985348105430603,
     'auc': 0.9967785626649857,
     'auc_precision_recall': 0.9990537911653519,
     'average_loss': 0.04917445499449968,
     'label/mean': 0.7985348105430603,
     'loss': 0.08832819480448961,
     'precision': 0.9909193366765976,
     'prediction/mean': 0.7993625551462173,
     'recall': 1.0,
     'global_step': 5000.0,
     'f1_score': 0.995436381378182}



```python
kfp.averages['test']
```




    {'accuracy': 0.9444444179534912,
     'accuracy_baseline': 0.800000011920929,
     'auc': 0.9571759253740311,
     'auc_precision_recall': 0.9839732944965363,
     'average_loss': 0.18561936542391777,
     'label/mean': 0.800000011920929,
     'loss': 0.18561936542391777,
     'precision': 0.9559257626533508,
     'prediction/mean': 0.8156909495592117,
     'recall': 0.9756944328546524,
     'global_step': 5000.0,
     'f1_score': 0.9655843100089102}



The seed of the best F1 score is stored in `best_seed`

```python
kfp.best_seed
```




    1852901353



## Documentation

To further inquire on the training parameters, how to store and load trained models, please refer to the [pipeline docs](/job_offer_classifier/pipeline_classifier). The validation method can be found in the [validations docs](/job_offer_classifier/validations) 
