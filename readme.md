# Automatic Library of Congress Classification Code

---

### Running the training code for non-sequential model

The main notebook for running all the models are in the notebook [].
Note that the training process required preprocessed embeddings data which lies in "github_data" folder. 
Please ensure that the data is downloaded and placed in the github_data folder.

The code for preprocessing these embeddings can be found in files under "runner" folder.


### Training code for sequential model

The training of LSTM on BERT embeddings were all done in Google Collab. 
These notebooks were then saved as jupyter notebook, and stored in this repository. 
To view the result, please view the notebooks in "report/rnn" folder.

The rnn codes (LSTM, GRU) can also be found in iml_group_proj/model/bert_[lstm|gpu].py
