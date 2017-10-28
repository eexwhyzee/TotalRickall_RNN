from utils import *
from model import *

#open text file with raw transcript, create dict for chars
content, chars, char_dict, int_dict = load_data(DATA_PATH)


# remove redundant sequences 
sentences, next_chars = remove_seq(content, SEQ_LENGTH)

# vectorization of sentences 

d_X, d_Y = sent2vec(sentences, SEQ_LENGTH, chars, char_dict, next_chars)

lstm = lstm_model(SEQ_LENGTH, chars)

checkpoint = ModelCheckpoint('saved_models/weights_{epoch:02d}-{loss:.4f}.hdf5',
                monitor='loss',
                verbose=1,
                save_best_only=True)

if TRAINING_MODE == True:
    temps = [0.2, 0.5, 1.0, 1.2]
    for iteration in range(1, ITERATIONS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        lstm.fit(d_X, d_Y,
                  batch_size=128,
                  epochs=1,
                  callbacks=[checkpoint])
        start_index = random.randint(0, len(content) - SEQ_LENGTH - 1)

        text_generator(temps,
                       SEQ_LENGTH,
                       TEXT_LENGTH,
                       chars,
                       char_dict,
                       int_dict,
                       lstm,
                       content)

else:
    lstm.load_weights(MODEL_WEIGHTS)

    text_generator(TEMPERATURE,
                   SEQ_LENGTH,
                   TEXT_LENGTH,
                   chars,
                   char_dict,
                   int_dict,
                   lstm,
                   content)

## TODO: Write generated text to txt file


