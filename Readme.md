# Textual Entailment and relatedness score

 In this project, deep learning models are used to predict the directional relation between 2 sentences, the possible relations are 'Entailment', 'Neutral', and 'Contradiction'. The data set used for the study is the SICK data set which can be downloaded from 'http://marcobaroni.org/composes/sick.html'.

 ### Methods
  <ul>
    <li>Bidirectional LSTM</li>
    <li>Siamese LSTM</li>
    <li>Siamese CNN</li>
    <li>GRU</li>
    <li>Deep RNN</li>
  </ul>
 <p><img src="arch_text_entailment.png" style="float:center" alt="drawing" width="500"/></p>

The highest accuracy was achieved using siamese LSTM for both entailment and relatedness task are as follows:
<ul>
    <li>Entailment Task: 84%</li>
    <li>relatedness Task: 81%</li>

## Getting Started

The models developed can be used for any similar tasks.<br>
Following section will help you clone and run this project in your local

### Prerequisites

### Prerequisites
Python version 3.7.7 was used for development.<br>
Python Packages required can be found in <i>'requirement.txt'</i><br>
The packages can be installed using the command:
```
pip install 'package_name'
```
Pretrained glove.6b.50d.txt word embedding was used in the deep learning models to assign weights, the word embedding file can be downloaded from https://nlp.stanford.edu/projects/glove/. The downloaded file must be placed in the word embedding folder, any other pretrained word embedding can be also be used.
