# Toxic-Comment-Detection
<img width="968" alt="Screenshot 2021-02-04 at 10 24 20 AM" src="https://user-images.githubusercontent.com/46114095/106846606-364bc500-66d3-11eb-977a-9e7e1f38cd2f.png">


## Problem statement -
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

## Evaluation Metric - 
Submissions are  evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

## Data - 
Data contains three files-
* train.csv which is supposed to be used to train the models.
* test.csv on which predictions should be made
* sample_submission.csv in which all the predictions should be store and submitted on kaggle.

Data contains columns like id,comment_text and 0 or 1 value corresponding to each of the category <br> like toxic, severe_toxic, obscene, threat, insult, identity_hate

# Approaches taken For Solving This Porblem-
* Logistic regression
* Naive Bayes
* Decision Trees
* LightGBM
* LSTM

# Results - 
I got the best results by Deep Learning LSTM approach. Followed by Ensemble of LightGBM and LSTM and LightGBM,Naive Bayes, Logistic regression, Decision Trees respectively.

## App Design Using Flask
I have also deployed my app using Flask and here is the view of it.
For non-toxic comments the predictions looked like this - 
<img width="1440" alt="Screenshot 2021-02-04 at 7 40 04 AM" src="https://user-images.githubusercontent.com/46114095/106846343-b4f43280-66d2-11eb-90f5-5cfd67e74e8d.png">
For some harsh comments the predictions looked like this - 
<img width="1440" alt="Screenshot 2021-02-04 at 9 27 37 AM" src="https://user-images.githubusercontent.com/46114095/106846464-ea991b80-66d2-11eb-9325-ef714784f09e.png">
