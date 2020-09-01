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
