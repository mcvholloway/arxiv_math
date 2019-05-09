#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, url_for, Markup
import fastText as ft
import pandas as pd
from scipy.sparse import load_npz
from functions.helper_functions import make_lookup_dict, ValuePredictor, FindSimilar, FindTags, preprocess_abstract
from functions.tagger import Tagger
from functions.ft_predictor import FTPredictor
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

#creating instance of the class
app = Flask(__name__)

tagger = Tagger()

loaded_model = ft.load_model('model_abstract.bin')

ft_predictor = FTPredictor(loaded_model)

with open('models/model_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('models/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

with open('../trained_models/tf_vect.pickle', 'rb') as pickle_file:
    tf_vect = pickle.load(pickle_file)
tfidf = load_npz('../trained_models/tfidf.npz')
df = pd.read_csv('../data/tagged.csv')
yake = pd.read_csv('../data/yake.csv')

#to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST'])
def result():
    global prediction
    global prediction_pre
    global to_predict_list
    global tags
    global mlb
    global lookup_dict
    global preprocessed
    global explainer
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))

        #preprocessed = preprocess_abstract(to_predict_list[0] + '. ' + to_predict_list[1])
        preprocessed = (to_predict_list[0] + '. ' + to_predict_list[1]).replace('\n', ' ')

        #prediction = ValuePredictor(preprocessed, pipeline, mlb, loaded_model)


        prediction_pre = ValuePredictor(preprocessed, ft_predictor, mlb)
        related_docs_indices, related_docs_similarities = FindSimilar(to_predict_list, tf_vect, tfidf)

        lookup_dict = make_lookup_dict()
        prediction = [lookup_dict[x] for x in prediction_pre]
        #prediction = ft_predictor.predict(to_predict_list[0] + '. ' + to_predict_list[1])
        #prediction = [x for x in mlb.inverse_transform(prediction)[0]]
        tags = tagger.generate_tags(to_predict_list[0] + ' ' + to_predict_list[1])

        #global prediction = prediction
        #global to_predict_list = to_predict_list

        explainer = LimeTextExplainer(class_names=list(mlb.classes_))

        return render_template("result.html",prediction=prediction, related_docs_indices = related_docs_indices, df = df, tags = tags, to_predict_list = to_predict_list, related_docs_similarities = related_docs_similarities, num = list(range(len(related_docs_indices))), lookup_dict = lookup_dict, prediction_pre = prediction_pre)

@app.route('/tag_finder', methods = ['POST'])
def tags():
    if request.method == 'POST':
        search = list(request.form.to_dict().values())[0]
        related_tags, related_tags_similarities = FindTags(to_predict_list, tf_vect, tfidf, search, df,yake)

        
        #return render_template("index.html")
        return render_template("result.html",prediction=prediction, related_docs_indices = related_tags, df = df, tags = tags, to_predict_list = to_predict_list, related_docs_similarities = related_tags_similarities, num = list(range(len(related_tags))), lookup_dict = lookup_dict, prediction_pre = prediction_pre)

@app.route('/AG')
def AG(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('AG'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/AT', methods = ['POST'])
def AT():
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('AT'),))
    html = exp.as_html()
    return render_template("AG.html", html = html)

@app.route('/AP')
def AP(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('AP'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/CT')
def CT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('CT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/CA')
def CA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('CA'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/CO')
def CO(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('CO'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/AC')
def AC(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('AC'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/CV')
def CV(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('CV'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/DG')
def DG(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('DG'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/DS')
def DS(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('DS'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/FA')
def FA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('FA'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/GM')
def GM(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('GM'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/GN')
def GN(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('GN'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/GT')
def GT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('GT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/GR')
def GR(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('AG'),))
    html = exp.as_html()
    return render_template("GR.html", html = Markup(html))

@app.route('/HO')
def HO(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('HO'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/IT')
def IT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('IT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/KT')
def KT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('KT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/LO')
def LO(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('LO'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/MP')
def MP(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('MP'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/MG')
def MG(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('MG'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/NT')
def NT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('NT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/NA')
def NA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('NA'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/OA')
def OA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('OA'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/OC')
def OC(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('OC'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/PR')
def PR(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('PR'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/QA')
def QA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('QA'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/RT')
def RT(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('RT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/RA')
def RA(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('RT'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/SP')
def SP(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('SP'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/ST')
def ST(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('ST'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))

@app.route('/SG')
def SG(): 
    exp = explainer.explain_instance(preprocessed, ft_predictor.predict_proba, num_features=6, labels = (list(mlb.classes_).index('SG'),))
    html = exp.as_html()
    return render_template("AG.html", html = Markup(html))


