# Abstract
The aim of this project is to experiment with Multi-Label text classification in legal documents. More specifically the dataset used is the EURLEX57K which consists of 57k legal documents, sourced from EU and annotated with aprox. 7k EUROVOC labels. Legal documents use domain specific language thus make them an interesting nlp task. In this project we will use a pre-trained BERT model to classify the documents. We will also experiment with different models and different ways of training the model. The results will be evaluated using the F1 score and the accuracy of predicting the labels. Also we will try to see the performance impact if we train the model with only the titles instead of the whole document. 

# Introduction
Text classification is a well established nlp task. It is used in many applications such as spam detection, sentiment analysis, topic classification, etc. In this project we will focus on multi-label text classification. In this task we have a set of documents and for each document we have a set of labels. The goal is to predict all the labels that are relevant for each document. Even more specifically the use of domain specific documents where large scale of labels are available for the classification make the task more complex. Multi Label text classification is being used in many applications such as news article classification, movie genre classification, medical records, etc. European Union has as it describes it a "multilingual and multidisciplinary thesaurus" called EUROVOC which consists of approx. 7k labels. These labels are manually being assigned to each legal document released by the EU. The dataset that we will use, EURLEX57k consists of 57k documents which are annotated with the EUROVOC labels. Out of the 7k labels only 4.3k have been assigned to at least one document of the dataset and a little more than 2k have been assigned to more than 10 documents. This makes the dataset a good candidate for also few shot learning and zero shot learning. Unfortunatelly this project will not focus on these tasks but is a good starting point for future work.



\section{Theory}

\subsection{Models}

\subsubsection{Multinomial Naive Bayes Classifier}

The Multinomial Naive Bayes Classifier is a probabilistic classifier based on Bayes Theorem. It is a simple and fast classifier which is often used as a baseline for text classification. It is based on the assumption that the features are independent. The classifier is trained by calculating the probability of each label given the document. The label with the highest probability is the predicted label. The formula for calculating the probability of a label given a document is:

\begin{center}
$
P(c$|$d) = \frac{P(c)\prod{P(w_{i}|c)^{f_{i}}}}{P(d)}
$
\end{center}

where $P(c|d)$ is the probability of a label $c$ given a document $d$, $P(d|c)$ is the probability of a document $d$ given a label $c$, $P(c)$ is the probability of a label $c$ and $P(d)$ is the probability of a document $d$. The probability of a document given a label is calculated by multiplying the probability of each word in the document. The probability of a word given a label is calculated by counting the number of times the word appears in documents with the label and dividing the result by the total number of words in documents. The probability of a label is calculated by counting the number of documents with the label and dividing the result by the total number of documents.  The MNB classifier is a good baseline for text classification because it is fast and simple. It is also a good choice for multi-label text classification because it can predict multiple labels for a document. The main disadvantage of the MNB classifier is that it assumes that the features are independent. This is not always the case in text classification. For example the word "not" is a negation word and it is often used in the context of a positive word. The MNB classifier will not be able to capture this relationship between the words.

\subsubsection{BERT}
BERT is an open-source pretrained model by GOOGLE which is based on Transformers. BERT originally is trained on the BookCorpus and the English Wikipedia. BERT relies on the Transformer architecture which is a deep learning model that is based on attention mechanism. The Transformer architecture is a stack of self-attention layers and feed-forward layers. The self-attention layers are used to capture the context of the words in the document. The feed-forward layers are used to capture the context of the sentences. This makes the model more robust and it can capture the context of the words and sentences in the document in comparison to traditional NLP methods like word embeddings, with big examples being GloVe and word2vec which map each word to a vector and do not take into account the context in which the word is in. The main disadvantage of the BERT model is that it is a large model and it requires a lot of computational resources to train it. In this project we will use LegalBERT which is a pretrained BERT model that is pretrained on legal documents. LegalBERT is based on the official BERT and has 12 layers with 768 hidden units, 12 heads and 110M parameters and is trained in legal documents from EU, UK and US. A graphical representation of the BERT model can be seen in the figure below.


\section{Data}
EURLEX57k is a dataset that consists of 57k documents which have been sourced and labeled by the EU Publication Office. The documents are in English and have been labeled with multiple concepts from EUROVOC, which is the thesaurus labels of EU regarding their deocuments. EUROVOC contains approximately 7k different keywords, organized in 21 domains and 127 sub-domains. Out of the 7k available concepts, only 4.3k have been assigned to at least one of the 57k documents. Moreovere only a litle over than 2k have been assigned to more than 10 documents. The average length of the body of each document is 547 words with a median of 399 and a maximum of 3479 words. The dataset is split into 3 subsets: train, validation and test. The train set contains 45k documents, the validation and test set contain 6k documents each.

\subsection{Data Preprocessing}
The preprocessing of the data is an important step in the training of the model. The preprocessing of the data is done in the following steps:

\begin{enumerate}
\item Remove stop words, punctuation, special characters and numbers and keep only the works.
\item Remove labels that have less than 10 documents.
\item Remove the documents that have no labels (because of the previous step).
\item Tokenize the documents with tf-idf for Naive Bayes and BERT tokenizer for BERT. 
\item Binary encode the labels.
\end{enumerate}

All the data we saved into disk after the lematization and labels filtering in two different datases. One including only titles and the other the body. The exact same steps were apllied to both versions.

\section{Training}
\subsection{Multinomial Naive Bayes Classifier}
The Multinomial Naive Bayes was trained with the use of MultiOutputClassifier from sklearn. The model was trained with the tf-idf vectorizer and the binary encoded labels. No special parameters were passed to the model.

\subsection{LegalBERT}
Since BERT has a limit of 512 tokens we truncated the documents to 512 tokens. The vectorizer used in this case is the BERT tokenizer. An extra Linear layer was added to the model to transform the output of the BERT model to the number of labels. The model was trained with the AdamW optimizer and the BCEWithLogitsLoss loss function. A batch size of 16 was used in both the body and titles datasets, with 300 epochs and learning rate of 2e-05. In the case of the titles dataset, Mixed Precision was used to speed up the training process, but in the case of the body dataset the GPU used, did not support it. The top 3 models were saved and the one with the best validation loss was used for the predictions. Both models had also early stopping with patience of 5 epochs. The training was done in a RTX 3070 8GB (title dataset since body dataset did not fit) and a GTX 1080TI 11GB (body dataset). Mixed precision significantly reduced the training time of the model.

\section{Evaluation}
For both models the evaluation is based on the F1-score and the accuracy. The F1-score is the harmonic mean of the precision and recall. The precision is the number of true positives divided by the number of true positives and false positives. The recall is the number of true positives divided by the number of true positives and false negatives. The accuracy is the number of true positives and true negatives divided by the total number of predictions. The evaluation was done with the use of the classification\_report from sklearn.metrics. The evaluation for Multinomial Naive Bayes Classifier was done on each label while for BERT the classes where converted to 0 and 1 based on a threshold of 0.4 which resulted in a better F1-score. The evaluation for both models was done on the test set.

\subsubsection{Body}
\begin{figure}[ht!]
    \centering
    \includegraphics[width=6cm]{mnb_body.png} 
    \caption{Classification Report for MNB with body data}
    \label{fig:MNB_body}
\end{figure}

\begin{figure}[ht!]
    \centering
    \includegraphics[width=6cm]{legal_bert_body.png} 
    \caption{Classification Report for LegalBERT with body data}
    \label{fig:legal_bert_body}
\end{figure}

\vskip 5in
\subsubsection{Titles}
\begin{figure}[ht!]
    \centering
    \includegraphics[width=6cm]{mnb_titles.png} 
    \caption{Classification Report for MNB with titles data}
    \label{fig:mnb_titles}
\end{figure}

\begin{figure}[ht!]
    \centering
    \includegraphics[width=6cm]{legal_bert_titles.png} 
    \caption{Classification Report for LegalBERT with titles data}
    \label{fig:legal_bert_titles}
\end{figure}

\newpage

\section{Discussion}

From the above figures we can easily notice how low the Multinomial Naive Bayes performed with a weighted average of 0.02 in both cases. On the other side, LegalBERT scored 0.83 in both scenarios. This comes with great surprice since the title data are way smaller and that resulted in faster training times. This could be based on the fact that BERT has the ability to learn the context of the words and the meaning of the sentences. This is a great advantage for the model since the documents are not that long and the model can learn the context of the words good enough as in the case of the body data. Moreover the pre-trained model has been trained on a big amount of legal data already which resulted in a better overall model for this specific task. The world of NLP is a fast evolving one and the legal domain can be benefited significantly from the advances in the field. Such models can be used to classify legal cases or even help the lawyers to find the right documents for their cases, solve legal issues or even help the judges to make better decisions and in shorter time. Another topic of that domain would be the summarization of such long documents. All these can be considered as future work.

\section{Conclusion}
This project intented to explore the use of NLP and Transformers that are pre-trained with legal data. The results seem promising and the models can be used in the future for other tasks in the legal domain. With the emerging advances in the field of NLP, the legal domain can be benefited significantly. The models from the above project can be further fine-tuned and used with bigger datasets. An interesting future work would be to modify the model and use it for the summarization of the documents. This would be a great help for the lawyers and judges to make better decisions and in shorter time.