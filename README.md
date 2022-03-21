# Qntfy categorization classifier

This is a multi-label logistic regression classifier that classifies
documents into categories. A similar model and training set generation
procedure was used for both World Modelers and for CauseEx. They will
be described separately below.

Both models are one vs. rest logistic regression classifiers that
allows for multiple labels; each probability score ranges from 0 to 1,
with scores higher to 1 reflecting a higher probability of the
document being classified into that category. Shorter documents will
tend to have lower scores, so if and when the app is modified to
export class predictions rather than probabilities, it will likely
make sense to lower the threshold (it defaults to .5).  

[![build and publish](https://github.com/twosixlabs-dart/qntfy-ner/actions/workflows/build-and-publish.yml/badge.svg)](https://github.com/twosixlabs-dart/qntfy-ner/actions/workflows/build-and-publish.yml)

## World Modelers

The model classifies each document into eight categories:

1. Agriculture
2. Climate
3. Commerce
4. Conflict
5. Economic
6. Health
7. Migration
8. Hydrology

It outputs probabilities for each of the eight categories. It was
trained on a set of 10,000 news articles, 1250 per class, and had an
F1 score of .979 when tested on 33% of the training data.

### WM training set

To generate the training set, we used a semi-supervised approach. Qntfy
hand labeled 160 documents - 20 per class - from the initial batch of
MITRE documents. Then tf-idf vectorized these 160 documents to train a
Linear SVC text classifier. Qntfy used this classifier to identify similar
articles by class with a dataset of articles from the NYT corpus (LDC2008T19) and
the AllTheNews corpus. Qntfy added the 30 documents with the highest
probability for classification per class and iterated through this
process - retraining the classifier with a larger dataset - until
10k documents was reached (1250 per class). See the notebook in this
repository (found at `notebooks/generating_data_set.ipynb`) for more
information.

### CE training set

To generate the training set (in the absensce of any sample documents),
we used a list of provided keywords for each category from
Two-Six. Grepped through the titles of the NYT dataset for these key
words and used them to assemble a preliminary set of training
documents for each category. Then followed the same procedure as was
done for the WM documents, iterating through keeping only documents
with a high probability of being classified within at least one
category until there were 1250 documents per class.

## Getting Started

### Install requirements
```
pip install -r requirements.txt
```

### Execution

Using the World Modelers model:
```
python app_WM.py
```

or, using the CauseEx model:

```
python app_CE.py
```

The pretrained model is stored in the directory and named `OvR_LR_model2.sav`.

Once you have input data, click "predict". This will output labeled
probabilities for the 8 categories.

## Delivery information

### Source Code

#### Service or integration code such as REST APIs or web applications

Contained in this repository:

- [World Modelers](./app_WM.py)
- [CauseEx](./app_CE.py)

#### Source code for model training

Contained in this repository, [here](./notebooks).

spacy: Available at the [spacy website][spacy-training].

### Models

#### Inventory of any open source / public models that were used

Only the models used for vectorizing the sentences are OSS/public.
They can be found at the [spacy website][spacy-models].

#### Information for how to obtain these models

Located in this repository:

- [WM](./model_WM_v0.sav)
- [CE](./model_CE_v0.sav)

spacy: Consult [this script](./dependencies.sh) or above site.

### Documentation

#### Reference information on the model or algorithm that is used for each analytic

See [above](./README.md).

spacy: Available at the [spacy website][spacy-models].

#### Documentation on how to train and deploy new models

See [here](./notebooks/generating_data_set.ipynb).

spacy: Available at the [spacy website][spacy-training].

#### Information on data cleaning, preparation, or formatting that is required for each model

- [World Modelers](./README.md#wm-training-set)
- [CauseEx](./README.md#ce-training-set)

spacy: Available at the [spacy website][spacy-models].

### Data

- [World Modelers](./README.md#wm-training-set)
- [CauseEx](./README.md#ce-training-set)

Additionally, download and extract
[this S3 file](s3://qntfy-artifacts/analytic_categorization.tgz)
for the files used in the training notebooks.

[spacy-training]: https://spacy.io/usage/training
[spacy-models]: https://spacy.io/models/en
