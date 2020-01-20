import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('./models/chatbot_120e.h5')
with open('./models/chatbot_token', 'rb') as f:
    tokenizer = pickle.load(f)


def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=156, max_question_len=6):
    X = []
    Xq = []
    Y = []

    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index)+1)

        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


def fix_spaces(s):
    new_string = ''
    for i in range(len(s)):
        if s[i] in '.,?' and s[i-1] != ' ':
            new_string += ' '
        new_string += s[i]
    return new_string


def get_answer(story, question):
    story = fix_spaces(story)
    question = fix_spaces(question)
    my_data = [(story.split(), question.split(), 'yes')]
    story, question, my_answer = vectorize_stories(my_data)
    pred_results = model.predict(([story, question]))
    val_max = np.argmax(pred_results[0])
    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    return k


message = 'Please use the following list of words'
print(message+'\n'+'-'*len(message))
print(', '.join([k for k, v in tokenizer.word_index.items()]))
print()

while True:
    my_story = input('Story > ')
    my_question = input('Question > ')
    print('Answer > ', get_answer(my_story, my_question))
