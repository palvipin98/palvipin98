#test run on sentimnet 3 
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Define Sentiment Analysis Model
class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)  # Output size 1 for binary sentiment classification

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Using [CLS] token's output for classification
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Define custom dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
def redundancy_analysis(text):
    # Example redundancy analysis code 
    word_count = len(text.split())
    unique_word_count = len(set(text.split()))

    # Calculate redundancy ratio
    if word_count == 0:
        redundancy_ratio = 0
    else:
        redundancy_ratio = (word_count - unique_word_count) / word_count

    # Assign score based on redundancy ratio
    if redundancy_ratio <= 0.2:
        score = 1
    elif 0.2 < redundancy_ratio <= 0.4:
        score = 2
    elif 0.4 < redundancy_ratio <= 0.6:
        score = 3
    elif 0.6 < redundancy_ratio <= 0.8:
        score = 4
    else:
        score = 5

    return score

def redundancy_analysis(text):
    # Example redundancy analysis code 
    word_count = len(text.split())
    unique_word_count = len(set(text.split()))

    # Calculate redundancy ratio
    if word_count == 0:
        redundancy_ratio = 0
    else:
        redundancy_ratio = (word_count - unique_word_count) / word_count

    # Assign score based on redundancy ratio
    if redundancy_ratio <= 0.2:
        score = 1
    elif 0.2 < redundancy_ratio <= 0.4:
        score = 2
    elif 0.4 < redundancy_ratio <= 0.6:
        score = 3
    elif 0.6 < redundancy_ratio <= 0.8:
        score = 4
    else:
        score = 5

    return score

def grammar_analysis(text):
    # Example grammar analysis code
    # This function can be implemented based on grammar rules or using a grammar checking tool/library
    # For simplicity, let's assume it returns a random score between 1 and 5
    import random
    return random.randint(1, 5)

def comprehension_analysis(text):
    # Example comprehension analysis code
    # This function can be implemented based on various factors such as sentence complexity, vocabulary level, etc.
    # For simplicity, let's assume it returns a random score between 1 and 5
    import random
    return random.randint(1, 5)

def relevance_analysis(text):
    # Example relevance analysis code
    # This function can be implemented based on the relevance of the text to a given topic or context
    # For simplicity, let's assume it returns a random score between 1 and 5
    import random
    return random.randint(1, 5)

def context_analysis(text):
    # Example context analysis code
    # This function can be implemented based on the context in which the text is used or presented
    # For simplicity, let's assume it returns a random score between 1 and 5
    import random
    return random.randint(1, 5)

def efficiency_analysis():
    
    import random
    return random.randint(1, 5)

def readability_analysis(text):
   
    import random
    return random.randint(1, 5)



def __len__(self):
        return len(self.texts)

def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Load and split dataset #ADD DATA here 
# For demonstration, let's assume you have your dataset in the form of lists: texts and labels
from sklearn.model_selection import train_test_split

data = [
    {
        'text': "This movie was great, i had so much fun this was a great night and i met the love of my live he is very beautiful and hadsomee boy, many boys fadeUnder the golden rays of the morning sun, the forest awakened with a symphony of chirping birds and rustling leaves. A gentle breeze danced through the branches, carrying with it the scent of pine and earth. The path meandered through the woods, inviting wanderers to explore its hidden treasures. As I walked, I couldn't help but feel a sense of peace and tranquility enveloping me. Every step brought me closer to nature's embrace, away from the hustle and bustle of the modern world. In this serene sanctuary, time seemed to stand still, allowing me to immerse myself fully in the beauty of the wilderness.",
        'sentiment': 1,  # Positive sentiment
        'parameters': {
            'Redundancy': 3,
            'Grammar': 4,
            'Comprehension': 5,
            'Relevance': 5,
            'Context': 4,
            'Accuracy': 4,
            'Efficiency': 3,
            'Readability': 4
        }
    },
    {
        'text': "The acting in this movie was terrible. The plot was confusing, and the dialogue was poorly written. I wouldn't recommend it to anyone.",
        'sentiment': 0,  # Negative sentiment
        'parameters': {
            'Redundancy': 2,
            'Grammar': 3,
            'Comprehension': 2,
            'Relevance': 3,
            'Context': 2,
            'Accuracy': 2,
            'Efficiency': 2,
            'Readability': 3
        }
    }
]

# Split into train and test sets
train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)

train_texts = [entry['text'] for entry in train_data]
train_sentiments = [entry['sentiment'] for entry in train_data]
train_parameters = [entry['parameters'] for entry in train_data]

test_texts = [entry['text'] for entry in test_data]
test_sentiments = [entry['sentiment'] for entry in test_data]
test_parameters = [entry['parameters'] for entry in test_data]

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = SentimentAnalysisModel()

# Define dataset and dataloader
MAX_LEN = 128  # Define your desired maximum sequence length
train_dataset = SentimentDataset(train_texts, train_sentiments, train_parameters, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_texts, test_sentiments,test_sentiments, tokenizer, MAX_LEN)

BATCH_SIZE = 16  # Define your desired batch size
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_epoch(model,dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# Train the model
NUM_EPOCHS = 3  # Define number of epochs for training

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}')

# Test example
def predict_sentiment(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        probability = torch.sigmoid(output).cpu().item()

        return probability
def calculate_composite_score(parameters):
    # Calculate the average score across all parameters
    total_score = sum(parameters.values())
    max_score = len(parameters) * 5  # Maximum possible score
    average_score = total_score / max_score
    print(average_score)
    print(calculate_composite_score)
    # Provide feedback based on the average score
    feedback = ""
    if average_score <= 2:
        feedback = "The text needs significant improvement in all aspects."
    elif average_score <= 3:
        feedback = "The text shows moderate performance in most aspects."
    elif average_score <= 4:
        feedback = "The text performs well in many aspects."
    else:
        feedback = "The text excels in all aspects."

    return feedback

# Example usage:
parameters = {
    'Redundancy': 3,
    'Grammar': 4,
    'Comprehension': 5,
    'Relevance': 5,
    'Context': 4,
    'Accuracy': 4,
    'Efficiency': 3,
    'Readability': 4
}

feedback = calculate_composite_score(parameters)
print("Feedback:", feedback)


#test your model
test_text = "I hate the movie."
sentiment_probability = predict_sentiment(model, tokenizer, test_text)
print("Sentiment Probability:", sentiment_probability)
redundancy_score = redundancy_analysis(test_text)
grammar_score = grammar_analysis(test_text)
comprehension_score = comprehension_analysis()
relevance_score = relevance_analysis()
context_score = context_analysis()
efficiency_score = efficiency_analysis()
readability_score = readability_analysis()

# Print analysis results with scores
print(f"Redundancy: {redundancy_score}")
print(f"Grammar: {grammar_score}")
print(f"Comprehension: {comprehension_score}")
print(f"Relevance: {relevance_score}")
print(f"Context: {context_score}")
print(f"Efficiency: {efficiency_score}")
print(f"Readability: {readability_score}")