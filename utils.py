import argparse
import numpy as np
import random
import sys

# Parsing command line arguments
arg_p = argparse.ArgumentParser()
arg_p.add_argument('-data_path', default='data/rm_trans.txt')
arg_p.add_argument('-temperature', type=float, default=1.0)
arg_p.add_argument('-seq_length', type=int, default=50)
arg_p.add_argument('-text_length', type=int, default=500)
arg_p.add_argument('-training_mode', type=bool, default=False)
arg_p.add_argument('-model_weights', default='')
arg_p.add_argument('-iterations', type=int, default=50)
args = vars(arg_p.parse_args())

DATA_PATH = args['data_path']
TEMPERATURE = args['temperature']
SEQ_LENGTH = args['seq_length']
TEXT_LENGTH = args['text_length']
TRAINING_MODE = args['training_mode']
MODEL_WEIGHTS = args['model_weights']
ITERATIONS = args['iterations']


def load_data(data_path):
    with open(data_path) as f:
        content = f.read()
    output = [item for item in content]

    chars = sorted(list(set(output)))
    char_dict = dict((c, i) for i, c in enumerate(chars))
    int_dict = dict((i, c) for i, c in enumerate(chars))

    return content, chars, char_dict, int_dict

def remove_seq(content, seq_length, step=3):
    step = step
    sentences = []
    next_chars = []
    for i in range(0, len(content) - seq_length, step):
        sentences.append(content[i: i + seq_length])
        next_chars.append(content[i + seq_length])
    return sentences, next_chars

def sent2vec(sentences, seq_length, chars, char_dict, next_chars):
    d_X = np.zeros((len(sentences), seq_length, len(chars)),
                   dtype=np.bool)
    d_Y = np.zeros((len(sentences), len(chars)),
                   dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            d_X[i, t, char_dict[char]] = 1
        d_Y[i, char_dict[next_chars[i]]] = 1

        return d_X, d_Y

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def text_generator(temperature,
                   seq_length,
                   text_length,
                   chars,
                   char_dict,
                   int_dict,
                   model,
                   content):

    start_index = random.randint(0, len(content) - seq_length - 1)
    for temp in [temperature]:
        print()
        print('----- Temperature:', temp)
        generated = ''
        sentence = content[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(text_length):
            x = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_dict[char]] = 1

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, temp)
            next_char = int_dict[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()



