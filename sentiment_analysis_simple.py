# NLTK is a natural language processing library
# Contains many useful corpus and algorithsm for language analysis
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# Function to create dictionary of words from corpus
def get_features(word_list):
    return dict([(word, True) for word in word_list])

# Create lists with positive and negative ids from movie_reviews database
positive_ids = movie_reviews.fileids('pos')
negative_ids = movie_reviews.fileids('neg')

# Get positive features from lists in dict
positive_features = [(get_features(movie_reviews.words(fileids=[f])),
           'Positive') for f in positive_ids]
negative_features = [(get_features(movie_reviews.words(fileids=[f])),
           'Negative') for f in negative_ids]

# Divide lists into train and test [0.75 train, 0.25 test]
threshold_positive = int(0.75 * len(positive_features))
threshold_negative = int(0.75 * len(negative_features))
features_train = positive_features[:threshold_positive] + negative_features[:threshold_negative]
features_test = positive_features[threshold_positive:] + negative_features[threshold_negative:]

# Train features on Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(features_train)
print "\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test)

# Sample input reviews
input_reviews = [
"John Wick is an incredible series!",
"Keanu Reeves has done an outstanding job.",
"The cinematography is pretty okay in this movie",
"The direction was different and the story was epic"
]
