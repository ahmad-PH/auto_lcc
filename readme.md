## Automatic Library of Congress Classification 

The [Library of Congress Classification (LCC)](https://www.loc.gov/catdir/cpso/lcco/) is a comprehensive classification system that was first developed in the late nineteenth and early twentieth centuries to organize and arrange the book collections of the Library of Congress. The vast complexity of this system has made manual book classification for it quite challenging and time-consuming. This is what has motivated research in automating this process, as can be seen in [Larson RR (1992)](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199203)43:2%3C130::AID-ASI3%3E3.0.CO;2-S?casa_token=n7OvACbIUskAAAAA:ipj3k3bbYhj5V9e7ZGjC8os77knWlKBUod9HVZTGESMIRw7YOjSAag1MIuGgaAEaceMYZo-w-GAgq5Q), [Frank and Paynter (2004)](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.10360?casa_token=OKy2l5FTC2IAAAAA:ZoULqLLnllGC-4e5JtVS48yVvXDcZfDiyWSYO51p6mSOxG1SBs9C4shX3GWsbi6e38BRERajygmNVz0), and [Ávila-Argüelles et al. (2010)](http://informatica.si/index.php/informatica/article/viewFile/277/273).

In this work we propose the usage of word embeddings, made possible by recent advances in NLP, to take advantage of the fairly rich semantic information that they provide. Usage of word embeddings allows us to effectively use the information in the synposis of the books which contains a great deal of information about the record. We hypothesize that the usage of word embeddings and incorporating synopses would yield better performance over the classifcation task, while also freeing us from relying on [Library of Congress Subject Headings (LCSH)](https://www.loc.gov/aba/publications/FreeLCSH/freelcsh.html), which are expensive annotations that previous work has used.

To test out our hypotheses we designed Naive Bayes classifiers, Support Vector Machines, Multi-Layer Perceptrons, and LSTMs to predict 15 of 21 Library of Congress classes. The LSTM model with large BERT embeddings outperformed all other models and was able to classify documents with 76% accuracy when trained on a document’s title and synopsis. This is competitive with previous models that classified documents using their Library of Congress Subject Headings.

For a more detailed explanation of our work, please see our [project report](https://github.com/ahmad-PH/auto_lcc/blob/main/report.pdf).

---
### Dependencies
To run our code, you need the following packages:
```
scikit-learn=1.0.1
pytorch=1.10.0
python=3.9.7
numpy=1.21.4
notebook=6.4.6
matplotlib=3.5.0
gensim=4.1.2
tqdm=4.62.3
transformers=4.13.0
nltk=3.6.5
pandas=1.3.4
seaborn=0.11.2
```

### Checklist

1. Install the python packages listed above with [requirements.txt](https://github.com/ahmad-PH/iml_group_proj/blob/main/requirements.txt)

```
$ pip install -r requirements.txt
```
or any other package manager you would like.


2. Set PYTHONPATH to the root of this folder by running the command below at the root directory of the project.

```
$ export PYTHONPATH=$(PWD)
```

3. Download the data needed from [this link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing) and put it in the project root folder. Make sure the folder is called `github_data`.

For the features (tf_idf, w2v, and BERT), you can also use the runner python scripts in "runner" folder to create features.

Use the command below to build all the features. The whole features preparation steps take around 2.5 hours.

```{shell}
$ python runner/build_all_features.py
```

Due to its large memory consumption, the process might crash along the way.
If that's the case, please try again by running the same command. The script is able to pick up on where it left of.

*Build each feature separately*

*BERT embeddings*

```{shell}
$ python runner/build_bert_embeddings.py --model_size=small  
```
*W2V embeddings*

For this one, you will need to run the `generate_w2v_embedddings.ipynb` notebook.

*tf-idf features*

```{shell}
$ python runner/build_tfidf_features.py
```

If the download still fails, then please download the data directly from our Google Drive [[Link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing)] (BERT small and large unavailable).


### Running the training code for non-sequential model

**Starting point**  
The main notebook for running all the models is in this notebook [[Link](https://github.com/ahmad-PH/iml_group_proj/blob/main/report/Library%20of%20Congression%20Classification.ipynb)].  
Note that the training process required preprocessed embeddings data which lies in "github_data" folder. 

**Caching**  
Note that once each model finishes fitting to the data, the code also stored the result model as a pickle file in the "_cache" folder.

![nonsequential_notebook_screenshot_1.png](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/nonsequential_notebook_screenshot_1.png?raw=true)


### Training code for sequential model

These notebooks for LSTM on BERT and word2vec ware all located in the `report/nnn` folder. (e.g., [[Link](https://github.com/ahmad-PH/iml_group_proj/blob/main/report/rnn/LibOfCongress_LSTM.ipynb)].


![screenshot_rnn_1](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_1.png?raw=true)

![screenshot_rnn_2](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_2.png?raw=true)


The rnn codes (LSTM, GRU) can also be found in iml_group_proj/model/bert_[lstm|gpu].py
