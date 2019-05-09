source venv/bin/activate

../fastText-0.2.0/fasttext supervised -input ../data/ftabstract_full.csv -output model_abstract -lr 0.5 -epoch 25 -wordNgrams 2 -bucket 200000 -dim 50 -loss one-vs-all

export FLASK_APP=script.py
flask run

