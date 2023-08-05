import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        super(DecoderRNN, self).__init__()

        self.LSTM = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True, 
                            batch_first=True,  
                            dropout=0,
                            bidirectional=False, 
                           )

        self.linear = nn.Linear(hidden_size, vocab_size)                     
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, features, captions):
        
        # Discard the <end> word
        captions = captions[:, :-1]     

        embeddings = torch.cat((features.unsqueeze(1), self.word_embeddings(captions)), dim=1)

        LSTM_out, _ = self.LSTM(embeddings)

        return self.linear(LSTM_out) 
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        SentenceWordsIndices = []
        
        for _ in range(max_len):
            
            LSTM_out, states = self.LSTM(inputs, states)
            
            ClassificationOutput = self.linear(LSTM_out.squeeze(1))
            
            MaxIndex = torch.argmax(ClassificationOutput)

            SentenceWordsIndices.append(int(MaxIndex.cpu().detach().item()))

            if MaxIndex == 1:  # <end>
                break
  
            inputs = self.word_embeddings(MaxIndex.unsqueeze(0)).unsqueeze(1)

        return SentenceWordsIndices