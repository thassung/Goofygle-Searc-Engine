from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
import numpy as np
from sklearn.neighbors import KDTree
import pickle

app = Flask(__name__, template_folder='templates')

class Skipgram(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(Skipgram,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(center_words) # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words) # [batch_size, 1, emb_size]
        all_embeds    = self.embedding_u(all_vocabs) #   [batch_size, voc_size, emb_size]

        scores      = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]

        nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
        # scalar (loss must be scalar)

        return nll # negative log likelihood

class SkipgramNegSampling(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_size) # out embedding
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words) # [batch_size, 1, emb_size]
        neg_embeds    = -self.embedding_u(negative_words) # [batch_size, num_neg, emb_size]

        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2))
        #[batch_size, k, emb_size] @ [batch_size, emb_size, 1] = [batch_size, k, 1]

        loss = self.logsigmoid(positive_score) + torch.sum(self.logsigmoid(negative_score), 1)

        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds

class GloVe(nn.Module):

    def __init__(self, vocab_size,embed_size):
        super(GloVe,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, embed_size) # out embedding

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        center_embeds = self.embedding_v(center_words) # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words) # [batch_size, 1, emb_size]

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #[batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        #note that coocs already got log
        loss = weighting*torch.pow(inner_product +center_bias + target_bias - coocs, 2)

        return torch.sum(loss)

class Search():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.word2index = {w: i for i, w in enumerate(vocab)}
        self.embeddings = [self.get_embed(word) for word in vocab[:-1]]
        self.kdtree = KDTree(self.embeddings)

    def get_embed(self, word):
        if word in self.vocab[:-1]:
            id_tensor = torch.LongTensor([self.word2index[word]])
            v_embed = self.model.embedding_v(id_tensor)
            u_embed = self.model.embedding_u(id_tensor)
            word_embed = (v_embed + u_embed) / 2
            x, y = word_embed[0][0].item(), word_embed[0][1].item()
            return np.array([x, y])
        else:
            return None

    def search(self, word):
        if word not in self.vocab:
            print('Unknown word')
            word_vec = np.array([0,0])
        else:
            print('Known word')
            word_vec = self.get_embed(word)
        _, indices = self.kdtree.query([word_vec], k=10)
        infer = [self.vocab[i] for i in indices[0]]
        return infer

model1 = torch.load('../model/Word2Vec_Skipgram.pth')
model2 = torch.load('../model/Word2Vec_Negative Sampling.pth')
model3 = torch.load('../model/GloVe.pth')
model4 = KeyedVectors.load('../model/GloVe_Gensim.model')

vocab_skipgram = pickle.load(open('../vocab/word2vec_vocab.pkl', 'rb'))
vocab_glove = pickle.load(open('../vocab/glove_vocab.pkl', 'rb'))

@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/search', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        searchData = request.json
        print(searchData)
        word = searchData['input'] 
        searchMode = int(searchData['searchEngine'])
        print(f'Search Mode: {searchMode}')
        print('>>>>> searching <<<<<')
        try:
            if searchMode == 1:
                bar = Search(model1, vocab_skipgram)
                result = bar.search(word)
            elif searchMode == 2:
                bar = Search(model2, vocab_skipgram)
                result = bar.search(word)
            elif searchMode == 3:
                bar = Search(model3, vocab_glove)
                result = bar.search(word)
            else:
                try: 
                    result = [w for w, _ in model4.most_similar(word)]
                except KeyError:
                    result = ['Unknown word ㅠㅠ']
            print(result)
            print('======= DONE =======')
            return result
        except:
            return ['Hmm... Something is not right.']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

