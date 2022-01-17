# Automatic Library of Congress Classification 

The [Library of Congress Classification (LCC)](https://www.loc.gov/catdir/cpso/lcco/) is a comprehensive classification system that was first developed in the late nineteenth and early twentieth centuries to organize and arrange the book collections of the Library of Congress. The vast complexity of this system has made manual book classification for it quite challenging and time-consuming. This is what has motivated research in automating this process, as can be seen in [Larson RR (1992)](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199203)43:2%3C130::AID-ASI3%3E3.0.CO;2-S?casa_token=n7OvACbIUskAAAAA:ipj3k3bbYhj5V9e7ZGjC8os77knWlKBUod9HVZTGESMIRw7YOjSAag1MIuGgaAEaceMYZo-w-GAgq5Q), [Frank and Paynter (2004)](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.10360?casa_token=OKy2l5FTC2IAAAAA:ZoULqLLnllGC-4e5JtVS48yVvXDcZfDiyWSYO51p6mSOxG1SBs9C4shX3GWsbi6e38BRERajygmNVz0), and [Ávila-Argüelles et al. (2010)](http://informatica.si/index.php/informatica/article/viewFile/277/273).

In this work we propose the usage of word embeddings, made possible by recent advances in NLP, to take advantage of the fairly rich semantic information that they provide. Usage of word embeddings allows us to effectively use the information in the synposis of the books which contains a great deal of information about the record. We hypothesize that the usage of word embeddings and incorporating synopses would yield better performance over the classifcation task, while also freeing us from relying on [Library of Congress Subject Headings (LCSH)](https://www.loc.gov/aba/publications/FreeLCSH/freelcsh.html), which are expensive annotations that previous work has used.

To test out our hypotheses we designed Naive Bayes classifiers, Support Vector Machines, Multi-Layer Perceptrons, and LSTMs to predict 15 of 21 Library of Congress classes. The LSTM model with large BERT embeddings outperformed all other models and was able to classify documents with 76% accuracy when trained on a document’s title and synopsis. This is competitive with previous models that classified documents using their Library of Congress Subject Headings.

For a more detailed explanation of our work, please see our [project report](https://github.com/ahmad-PH/auto_lcc/blob/main/report.pdf).

## Getting Started
1. First, you will need to install the dependencies. You can either install the packages manually from the list below:
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

Or, use pip to install them automatically:

```
$ pip install -r requirements.txt
```


2. Set PYTHONPATH to the root of this folder by running the command below at the root directory of the project.

```
$ export PYTHONPATH=$(PWD)
```

3. Download the data needed from [this link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing) and put it in the project root folder. Make sure the folder is called `github_data`. This folder contains training and test data, and most of the needed features.

4. Use the command below to generate BERT embeddings:
```{shell}
$ python core/generate_embeddings/build_bert_embeddings.py --model_size=small  
```

* Note: You can also use the command below to build all the features: 

```{shell}
$ python core/generate_embeddings/build_all_features.py
```

The whole features preparation steps take around 2.5 hours.

Due to its large memory consumption, the process might crash along the way.
If that's the case, please try again by running the same command. The script is able to pick up on where it left of.

### Running the training code for non-sequential models

**Starting point**  
The main notebook for running all the models is in this notebook: `notebooks/Library of Congress Classification.ipynb`.
Note that the training process required preprocessed embeddings data which lies in "github_data" folder. 

**Caching**  
Note that once each model finishes fitting to the data, the code also stored the result model as a pickle file in the "_cache" folder.

### Training code for sequential models

Notebooks for LSTM on BERT and word2vec are all located in the `notebooks/rnn` folder.

The rnn codes (LSTM, GRU) can also be found in core/model/bert_[lstm|gpu].py

## Results
The below table contains a summary of algorithm performances based on the choice of features.
|            	| Naive Bayes 	| SVC    	| MLP    	| LSTM   	|
|------------	|-------------	|--------	|--------	|--------	|
| Accuracy   	|             	|        	|        	|        	|
| TF-IDF     	| 33.87%      	| 62.30% 	| 67.93% 	| NA     	|
| word2vec   	| 46.07%      	| 69.03% 	| 69.93% 	| 71.77% 	|
| BERT_large 	| 50.67%      	| 75.13% 	| 75.60% 	| 76.03% 	|
| Precision  	|             	|        	|        	|        	|
| TF-IDF     	| 42.25%      	| 66.99% 	| 68.50% 	| NA     	|
| word2vec   	| 51.69%      	| 69.60% 	| 71.12% 	| 72.12% 	|
| BERT_large 	| 53.57%      	| 75.43% 	| 75.79% 	| 76.10% 	|
| Recall     	|             	|        	|        	|        	|
| TF-IDF     	| 33.87%      	| 62.30% 	| 67.93% 	| NA     	|
| word2vec   	| 46.07%      	| 69.03% 	| 69.93% 	| 71.77% 	|
| BERT_large 	| 50.67%      	| 75.13% 	| 75.60% 	| 76.03% 	|
| F1 Score   	|             	|        	|        	|        	|
| TF-IDF     	| 33.82%      	| 62.59% 	| 67.97% 	| NA     	|
| word2vec   	| 46.45%      	| 69.13% 	| 69.83% 	| 71.75% 	|
| BERT_large 	| 50.38%      	| 75.19% 	| 75.51% 	| 76.01% 	|


## Contributors (in no specific order)

* **Katie Warburton** - *Researched previous automatic LCC attempts and found the dataset. Wrote the introduction and helped to write the discussion. Researched and understood the MARC 21 bibliographic standard to parse through the dataset and extract documents with an LCC, title, and synopsis. Balanced the dataset and split it into a train and test set. Described data balancing and the dataset in the report.* - [katie-warburton](https://github.com/katie-warburton)

* **Yujie Chen** - *Trained and assessed the performance of SVM models and reported the SVM and general model development approaches and relevant results.* - [Yujie-C](https://github.com/Yujie-C)

* **Teerapat Chaiwachirasak** - *Wrote the code for generating tf-idf features and BERT embeddings. Trained Naive Bayes and MLP on tf-idf features and BERT embeddings. Wrote training pipelines that take ML models from the whole team and train them together in one same workflow with multiple data settings (title only, synopsis only, and title + synopsis) to get a summarized and unified result. Trained LSTM models on BERT embeddings on (Google Collab).* - [teerapat-ch](https://github.com/teerapat-ch)

* **Ahmad Pourihosseini** - *Wrote the code for generating word2vec embeddings and its corresponding preprocessing and the code for MLP and LSTM models on these embeddings. Came up with and implemented the idea of visualizing the averaged embeddings. Wrote the parts of the report corresponding to these sections.* - [ahmad-PH](https://github.com/ahmad-PH)

