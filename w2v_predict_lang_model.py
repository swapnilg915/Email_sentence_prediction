import pickle, traceback
from keras.models import load_model
from numpy import array
import time
from keras.preprocessing import sequence


class predictText():
    
    def __init__(self):
        self.max_length = 3
        self.num_of_words_to_predict = 8
        self.model, self.tokenizer = self.load_model()
        print("\nloaded model === ")
        
        
    def load_model(self):
        try:
            model = load_model('w2v_lang_model_2_1.h5')
            tokenizer = pickle.load(open('w2v_tokenizer_lang_model_2_1_2.pkl', 'rb'))

        except Exception as e:
            print ("\n error in model loading = ",e,"\n ",traceback.format_exc())
        return model, tokenizer
    
    # generate a sequence from the model
    def generate_seq(self, seed_text, max_len):
        n_words = self.num_of_words_to_predict
        in_text = seed_text
        for _ in range(n_words):
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = sequence.pad_sequences([encoded], maxlen=max_len, padding='pre')
            yhat = self.model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word, index in self.tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            in_text += ' ' + out_word
        return in_text
    
    def main(self):
        # evaluate
        print(self.generate_seq('nice talking', self.max_length-1))
        print(self.generate_seq('nice talking to', self.max_length-1))
        print(self.generate_seq('please let me know when', self.max_length-1))
        
        
    
if __name__ == '__main__':
    
    obj = predictText()
    obj.main()
    
