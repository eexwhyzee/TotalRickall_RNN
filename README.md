# Character RNN with Keras 

## Overview

Using a LSTM RNN to generate text conversations from the TV show [Rick and Morty](http://www.imdb.com/title/tt2861424/).

## Model

The model used is a 2-layer LSTM RNN with 784 neurons in each layer followed by a final dense output layer with a softmax activation. The model was trained the using [episode transcripts](http://rickandmorty.wikia.com/wiki/Category:Transcripts). More details on the model setup can be found in the `model.py` file. In addition, I've also implemented the use of Temperature (found in the `utils.py` file) when sampling text characters with the softmax, where lower temperature will create text output that will be more conservative and generally have less mistakes and more repetitive patterns. Meanwhile, higher temperature will create outputs that have more diverse text with the higher risk of mistakes.   

## Usage

Install the required packages:

```
pip install requirements.txt
```

Clone the repo onto your machine:
```
git clone https://github.com/eexwhyzee/TotalRickall_RNN.git
cd TotalRickall_RNN/
```

If you would like to train a model from scratch, you can run the `rm_text_generator.py` script with the argument `-training_mode` set to `TRUE` and the `-data_path` argument set to path of your text data you would like to train with. In addition, you can set the number of training iterations using the `-iterations` argument (default is `50`). After each training iteration, a sample text will be outputted in your terminal with varying temperatures (`0.2, 0.5, 1.0, 1.2`).

Otherwise, if you would like to start generating text right away with the trained model weights that I've included in this repo (trained on gcloud compute engine with Nvidia Tesla K80 GPU for 73 epochs), just run the `rm_text_generator.py` script with the path of the trained model weights and the length of text (`-text_length`) you wish to be generated (default is `500`):

```
python rm_text_generator.py -model_weights saved_models/weights_rm_lstm_v3_0.1199.hdf5
```

Here's a cleaned up (fixing spelling errors) sample of the generated text with the trained model weights:
```
Morty: What? W-what are you doing what?! W-W-W-W-Where---I-I-I-I-I-I-I-I-I'm the world is that problem to have to be remensive you feet to be a strange, day, but whales that dimension on phasing children on this one. And you want to go any horse.
Morty: Why does it here?

Doofus Rick: Hold on. We will respect alien fan...
Morty: What are you talking about?

Rick: You know, Morty, I’ve got a little stalled by a rumble! I want to go to the carpet. Thank you, a terror, with the rest of us a little bit of the brain bulls and begins next to the living room watching TV.

Morty: What?!
Rick: It's my first busting the dumb op, man. Come on, Morty [burps]! Presses the head water follow schooling and consider it and goes in a hellish butts to his dream where a god in Blip Blam the kitchen, and fall to go.
Beth: Damn it! Don’t go!

Poncho: *laughing* I… anyways advertise up in the come where he’s carrying and well be relax. It stops her life away, like you’re some to visiter than out of smell moderation and don't up. (Rick portals the room full of the portal gun his last part of the popeye police into a hots.)

Morty (C-137): Oh, my god! How can you like to ride a dick move the kitchen, the rembember is him to figure of themplitine. It's a metal existing.

Vick: Alright, merry, I don't want to know this, but I can’t amount herpie it up.

Morty: Uh-huh, you know what?! It’s not a plan, Jerry. What are you doing in here?

Jerry: What the hell?

All Mortys: What's going on?

Beth: What? Just kill on the phone, what do you want connect on the moniter, but the served directions, pens. What do you want for pulling the rust these this turn it out?

Birdperson: Yes.

Jerry: What the hell? We care the mets on the surprise coming inside and killing them to plug Morty. Don't even know this again?

Jerry: Summer, Morty don’t judge. I didn't know if It's a look where cares of Pickle Rick.

Pickle Rick: Why don't we up. This is a family as No. Do Earth -- but the soldiers from down on Jerry, they’re about to his bud launch bend, and then they they they they can bake up at the continues. I don't really an a possible. I want you to at them, where are we gonna go time the megabling can be my learn.
```



## Resources
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Automated Crowdturfing Attacks and Defenses in Online Review Systems](https://arxiv.org/abs/1708.08151)
- [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
- [LSTM Text Generation Example from Keras](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
