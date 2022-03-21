import sys
import os
import logging
import json
from typing import Dict

from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


with open('model_WM_v0.sav', 'rb') as f:
    mlb, LR_pipeline = pickle.load(f)

#flask stuff
app = Flask(__name__)

STATUS_OK = 'healthy'


def facet_annotation_dict() -> Dict:
    """Returns a skeleton dictionary for use in CDR output."""
    cat = {}
    cat['type'] = "facets"
    cat['label'] = "Qntfy Categorization"
    cat['version'] = "0.3.0"
    cat['class'] = 'derived'
    return cat


def text_to_categories(txt: str) -> Dict[str, float]:
    """Takes in a string of text, and predicts categories of the text.

    Returns output in a dictionary strings, which represent the categories,
    to floats, which represent their probabilities."""
    massaged = [txt]
    raw_outputs = LR_pipeline.predict_proba(massaged)[0]
    logging.debug(raw_outputs)
    return dict(zip(mlb.classes_, raw_outputs))


def cdr_to_cdr_categories(cdr: Dict) -> Dict:
    """Given a CDR-like dictionary, run categorization over extracted_text field
    and return predicted categories.

    Errors if the key extracted_text is not in the input dictionary."""
    txt = cdr['extracted_text']
    logging.debug('incoming text: {}'.format(txt))
    outputs = text_to_categories(txt)

    # massage into CDR schema
    category_dict = {}
    category_dict['label'] = outputs
    pairs = [{"value": cat, "score": proba} for cat, proba in outputs.items()]
    outdict = facet_annotation_dict()
    outdict['content'] = pairs
    return outdict


@app.route('/api/v1/health', methods=['GET'])
def health():
    out = {}
    out['status'] = STATUS_OK
    return jsonify(out)


@app.route('/api/v1/annotate/cdr',methods=['POST'])
def predict():
    """Like /predict, but only returns the annotation, not the entire document."""
    request_json = request.get_json(force=True)
    cats = cdr_to_cdr_categories(request_json)
    return jsonify(cats)


@app.route('/api/v1/cdr/predict',methods=['POST'])
def predictFull():
    """Returns the original CDR document with an added annotation for the predicted categories."""
    request_json = request.get_json(force=True)
    cats = cdr_to_cdr_categories(request_json)

    annos = []
    if 'annotations' in request_json:
        annos = request_json['annotations']
    annos.append(cats)
    request_json['annotations'] = annos
    return jsonify(request_json)


if __name__ == '__main__':
    app.run(debug=False,
            host='0.0.0.0',
            use_reloader=False,
            threaded=True,
            )
