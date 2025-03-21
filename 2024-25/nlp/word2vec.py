import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm



class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        # Inizializzazione delle matrici di embedding per input e output
        self.in_embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.out_embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        
    def forward(self, center_word_idx):
        # Ottieni l'embedding per la parola centrale (lookup manuale)
        center_embedding = self.in_embedding[center_word_idx]  # (batch_size, embedding_dim)
        # Calcolo del prodotto scalare con tutti gli embeddings del contesto
        scores = torch.matmul(center_embedding, self.out_embedding.T)  # (batch_size, vocab_size)
        return scores
    
    def train(self, dataloader, n_epochs: int, step: float = 0.1):
        history = []
        check = int(n_epochs * step)
        for epoch in range(n_epochs):
            total_loss = 0
            for center, context in dataloader:
                # Reset dei gradienti
                self.optimizer.zero_grad()
                # Forward pass
                output = self(center)
                # Calcolo della perdita
                loss = self.criterion(output, context)
                # Backpropagation
                loss.backward()
                # Ottimizzazione
                self.optimizer.step()
                total_loss += loss.item()
            if epoch % check == 0:
                history.append(total_loss)
        return history
    
    @property
    def embeddings(self):
        return self.in_embedding.data
    
    def predict_context_words(self, word, vocabulary, top_k=10):
        # Converti la parola in indice
        word_idx = vocabulary.word2idx([word])[0]
        word_tensor = torch.tensor([word_idx], dtype=torch.long)        
        # Usa il modello per ottenere l'output (punteggi)
        with torch.no_grad():  # Disabilitiamo il calcolo dei gradienti per la predizione
            output_scores = self(word_tensor)  # (1, vocab_size)
        # Applica softmax per ottenere la distribuzione di probabilità
        probs = F.softmax(output_scores, dim=1)  # (1, vocab_size)
        # Trova le parole con la probabilità più alta
        top_probs, top_indices = torch.topk(probs, top_k)  # Prendi le top_k parole
        top_indices = top_indices[0].tolist()  # Converte gli indici in una lista
        # Converti gli indici delle parole in testo
        predicted_words = vocabulary.idx2tokens(top_indices)
        predicted_probs = top_probs[0].tolist()
        return predicted_words, predicted_probs


class Word2WordPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Word2WordPrediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.vectors = np.zeros((input_dim, hidden_dim))
    
    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        output = F.softmax(output, dim=-1)
        return output
    
    def train(self, data_loader, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        
        # Ciclo di allenamento
        run = list(range(epochs))
        for epoch in tqdm(run):
            running_loss = 0.0
            for inputs, labels in data_loader:
                # Resettiamo i gradienti dell'ottimizzatore
                optimizer.zero_grad()
                # Forward pass: otteniamo le predizioni
                outputs = self(inputs)
                # Calcoliamo la loss
                loss = criterion(outputs, labels)
                # Backward pass: calcoliamo i gradienti
                loss.backward()
                # Aggiorniamo i pesi
                optimizer.step()
                # Aggiorniamo la loss per il batch corrente
                running_loss += loss.item()
            
            # Stampiamo la loss media per ogni epoch
            history.append(running_loss/len(data_loader))
        self._embeddings()
        return history
    
    def get_vector(self, word_idx: int):
        return self.vectors[word_idx]

    def _embeddings(self):
        weights_fc1 = self.fc1.weight.data.detach().numpy()
        self.vectors = weights_fc1.T
