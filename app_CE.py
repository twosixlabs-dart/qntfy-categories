import sys
import os
import logging
import json

from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd
import joblib
import numpy as np
import spacy

# shape of output data
cat = {}
cat['type'] = "facets"
cat['label'] = "Qntfy Categorization"
cat['version'] = "0.3.0"
cat['class'] = 'derived'


# Load spacy model for pooled embedding features
vectorizer = spacy.load('en_core_web_lg')
logging.debug('language model loaded')

# Load model
model_path = os.getenv('MODEL_PATH')
if (model_path is None) or (model_path == ''):
    model_path = 'expanded_cat_model.joblib'

model = joblib.load(model_path)
logging.debug('category model loaded')

# Hack - label names
label_names = ['political',
                'military',
                'social',
                'economic',
                'infrastructure',
                'information',
                'economic_production',
                'institutions',
                'leadership',
                'transportation',
                'economic_distribution',
                'economic_consumption',
                'economic_informality',
                'global_info',
                'affinity_groups',
                'political_org',
                'central_gov',
                'military_ind_base',
                'climate',
                'weather',
                'sustainment',
                'utilities',
                'national_info',
                'armed_forces',
                'habitability',
                'military_short',
                'economic_performance',
                'internal_security',
                'basic_needs',
                'defense_infrastructure',
                'phys_environment']

#flask stuff
app = Flask(__name__)

STATUS_OK = 'healthy'


# Utility function for converting log-probs to probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@app.route('/api/v1/health', methods=['GET'])
def health():
    out = {}
    out['status'] = STATUS_OK
    return jsonify(out)


@app.route('/api/v1/annotate/cdr',methods=['POST'])
def predict():
    request_json = request.get_json(force=True)
    txt = request_json['extracted_text']
    logging.debug('incoming text: {}'.format(txt))
    cat_classes = []
    category_dict = {}
    x_input = np.expand_dims(vectorizer(txt).vector, 0)
    model_scores = np.squeeze(sigmoid(model.decision_function(x_input))).tolist()
    logging.debug(model_scores)
    outputs = dict(zip(label_names, model_scores))
    logging.info(outputs)
    category_dict['label'] = outputs
    pairs = [{"value": cat, "score": proba} for cat, proba in outputs.items()]
    cat_classes.append(pairs)
    [cat['content']] = cat_classes
    return jsonify(cat)


if __name__ == '__main__':
    app.run(debug=False,
            host='0.0.0.0',
            use_reloader=False,
            threaded=True,
            )
