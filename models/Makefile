train.csv:
	python scripts/make_train_test.py

train_res.csv: train.csv
	python scripts/make_res.py

tf_vect.pickle: train.csv
	python scripts/tf.py

tf_vect_res.pickle: train_res.csv
	python scripts/tf_resample.py

ft_train.csv: train.csv
	python scripts/ft_train.py

ft_model.bin: ft_train.csv
	../fastText-0.2.0/fasttext supervised -input ft_train.csv -output ft_model -lr 0.5 -epoch 25 -dim 50 -loss one-vs-all 

ft_train_res.csv: train_res.csv
	python scripts/ft_train_res.py

ft_model_res.bin: ft_train_res.csv
	../fastText-0.2.0/fasttext supervised -input ft_train_res.csv -output ft_model_res -lr 0.5 -epoch 25 -dim 50 -loss one-vs-all

model_pipeline.pkl:
	python scripts/make_final_model.py

full_ft_train.csv:
	python scripts/make_res_full.py

full_ft_model.bin: full_ft_train.csv
	../fastText-0.2.0/fasttext supervised -input full_ft_train.csv -output ft_model -lr 0.5 -epoch 25 -dim 50 -loss one-vs-all 

ft_vectors.csv:
	python scripts/make_ft_vectors_df.py

ft_vectors.bin: ft_vectors.csv
	../fastText-0.2.0/fasttext skipgram -input ft_vectors.csv -output ft_vectors -dim 100 -thread 4 

