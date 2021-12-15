## Automatic Library of Congress Classification Code

---

### Checklist

1. Install python packages with [requirements.txt](https://github.com/ahmad-PH/iml_group_proj/blob/main/requirements.txt)

```
$ pip install -r requirements.txt
```

2. Ensure that there are data in "github_data" 

If not, use the runner python scripts in "runner" folder to create features.

*BERT embeddings*

```{shell}
$ python runner/build_bert_embeddings.py --model_size=small  
```

Note that as the whole process requires large amount of memory, the process might crash halfway.
If that's the case, please try again by running the same command. The script is able to pick up on where it left of.

*Word2Vec embeddings*

Please download directly from our Google Drive. [[Link](https://drive.google.com/drive/folders/1B-XNvIdGZazLvDjnH2xWGUBfoe-Jt53B?usp=sharing)]
<!-- ```{shell} -->
<!-- $ python runner/build_bert_embeddings.py --model_size=small -->  
<!-- ``` -->


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

The training of LSTM on BERT embeddings were all done in Google Collab. 
These notebooks were then saved as jupyter notebook, and stored in this repository. 
To view the result, please view the notebooks in "report/rnn" folder (e.g., [[Link](https://github.com/ahmad-PH/iml_group_proj/blob/main/report/rnn/LibOfCongress_LSTM.ipynb)].


![screenshot_rnn_1](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_1.png?raw=true)

![screenshot_rnn_2](https://github.com/ahmad-PH/iml_group_proj/blob/main/public/rnn_notebook_screenshot_2.png?raw=true)


The rnn codes (LSTM, GRU) can also be found in iml_group_proj/model/bert_[lstm|gpu].py
