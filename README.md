# Sentiment Classifier
> Classification of email job offer response emails 


This project classifies job offer response emails as 'positive' or 'negative' according to whether an email response expresses an interest in a job offer. The dataset contains job offer response emails annotated with 'positive' and 'negative' labels. The positive labels represent an interest in a job offer.

## Install

Clone the [GitHub](https://github.com/kikejimenez/job_offer_classifier) repository and `cd` to the cloned repo directory. Install the depencies in the `requeriments.txt`. The use of *REPL* or  *Notebook* is strongly  recommended

## How to use

### Run the Pipeline

First load and run the data science pipeline by importing the library  *job_email_classifier.pipeline_classifier*

```
from job_offer_classifier.pipeline_classifier import Pipeline
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-283fc65414a6> in <module>
    ----> 1 from job_offer_classifier.pipeline_classifier import Pipeline
    

    ModuleNotFoundError: No module named 'job_offer_classifier'


Instantiate the class `Pipeline` and call the `pipeline` method. This method loads the dataset, and trains and evaluates the model. The source file is the annotated dataset of payloads.

```
pl = Pipeline(src_file = '../data/interim/payloads.csv')
pl.pipeline()
```

### Predict Job Offer Sentiments

To make a prediction, use the `sentiment` method

```
pl.sentiment(''' Thank you for offering me the position of Merchandiser with Thomas Ltd.
I am thankful to accept this job offer and look ahead to starting my career with your company
on June 27, 2000.''')
```




    'positive'



One can take an example from the test set, contained in the `dfs` attribute. This attribute is a dictionary of  pandas dataframes.

```
example = pl.dfs['test'].sample().payload.iloc[0]
print(example.strip())
```

    good morning.
    thanks for your e-mail.
    please find enclosed resume for your kind perusal.


```
pl.sentiment(example)
```




    'positive'



## Performance

We use two tools to assesss the performance of the model:
  - Confusion Matrix 
  - K fold Validation

### Confusion matrix

To plot the confusion matrix the `Pipeline` has the method `plot_confusion_matrix`.

```
pl.plot_confusion_matrix('train')
```


![png](docs/images/output_19_0.png)


```
pl.plot_confusion_matrix('test')
```


![png](docs/images/output_20_0.png)


The 7% of the cases that are negatives are considered as positive. Less than one percent of the cases that are positive are considered as negative. Therefore, the model tends to be benevolent regarding job acceptances. This is consistent with that fact that the dataset has more positive cases than negative cases

### K fold validation

To assess the performance of the model via the k fold validation method, import the class `KFoldPipeline` in the  *k_fold_validation* submodule of *job_offer_classifier*

```
from job_offer_classifier.k_fold_validation import KFoldPipeline
```

Run the `k_fold_validation` method

```
kfp = KFoldPipeline(dataset_file='../data/interim/payloads.csv',n_splits=4)
kfp.k_fold_validation()
```

The averaged scores are stored in `avg_evaluation`

```
kfp.avg_evaluation['train']
```




    {'accuracy': 0.9889840483665466,
     'accuracy_baseline': 0.7989590167999268,
     'auc': 0.9948678016662598,
     'auc_precision_recall': 0.9983210116624832,
     'average_loss': 0.05849529942497611,
     'label/mean': 0.7989590167999268,
     'loss': 0.045953736174851656,
     'precision': 0.9893090277910233,
     'prediction/mean': 0.7996942400932312,
     'recall': 0.9964020103216171,
     'global_step': 5000.0,
     'f1_score': 0.992838505068065}



```
kfp.avg_evaluation['test']
```




    {'accuracy': 0.9450244158506393,
     'accuracy_baseline': 0.7994505614042282,
     'auc': 0.4663415998220444,
     'auc_precision_recall': 0.9501921236515045,
     'average_loss': 0.16826009936630726,
     'label/mean': 0.7994505614042282,
     'loss': 0.16826009936630726,
     'precision': 0.9409254342317581,
     'prediction/mean': 0.8031759560108185,
     'recall': 0.9670821875333786,
     'global_step': 5000.0,
     'f1_score': 0.9534101383560495}



Over 4 foldings, the averaged accuracy of 94%, while the F1 score is 95%

## Documentation

To inquire more on the training parameters, how to store and load trained models, please refer to the [pipeline_classifier](/job_offer_classifier/pipeline_classifier/) module. The k fold validation can be found in the [k_fold_validation](/job_offer_classifier/k_fold_validation/) module
