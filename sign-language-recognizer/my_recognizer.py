import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # DONE implement the recognizer
    for index in range(test_set.num_items):
        top_prob, top_word = None, None
        i_word_probabilities = {}
        # Get current test set sequences
        i_sequences, i_lengths = test_set.get_item_Xlengths(index)
        for word, model in models.items():
            try:
                # Calculate test score
                i_word_probabilities[word] = model.score(i_sequences, i_lengths)
            except Exception as e:
                i_word_probabilities[word] = float("-inf")
            # Update best probability if required
            if(top_prob == None or i_word_probabilities[word] > top_prob):
                top_prob, top_word = i_word_probabilities[word], word
            continue
        probabilities.append(i_word_probabilities)
        guesses.append(top_word)
    return probabilities, guesses
