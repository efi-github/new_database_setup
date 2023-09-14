import time

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import torch
import tqdm
import numpy as np
import nltk
from utilities import Timer
nltk.download('reuters')  # Download the dataset
from nltk.corpus import reuters
import nltk
nltk.download('punkt')
import nltk
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def initialize():
    sentences = reuters.sents()
    sentences = [" ".join(sent) for sent in sentences]
    return sentences


def initialize_movie_reviews():
    sentences = []
    for fileid in tqdm.tqdm(movie_reviews.fileids()):
        # get all sentences in the file
        temp_sentences = movie_reviews.sents(fileid)
        # make strings out of every sentence in the file
        temp_sentences = [" ".join(sent) for sent in temp_sentences]
        sentences.extend(temp_sentences)
    return sentences

def collate_fn(batch):
    input_ids, attention_mask = zip(*batch)
    input_ids = [torch.tensor(x) for x in input_ids]
    attention_mask = [torch.tensor(x) for x in attention_mask]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return input_ids, attention_mask

def normal(sentences, batch_size=2):
    print("Normal batching")
    print(f"Batch size: {batch_size}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Tokenize sentences at once and convert to NumPy
    current_time = time.time()
    tokenized_output = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    last_time = current_time
    current_time = time.time()
    print(f"Tokenizing took {current_time - last_time} seconds.")

    dataset = TensorDataset(tokenized_output["input_ids"], tokenized_output["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Initialize BERT model and set it to evaluation mode
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Loop through DataLoader to get batches
    for i, (batch_input_ids, batch_attention_mask) in enumerate(tqdm.tqdm(dataloader)):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            embeddings = outputs.last_hidden_state
            #print(f"Batch {i + 1} embeddings shape: {embeddings.shape}")


def sorted_dyn_batch(sentences, batch_size=2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Tokenize sentences at once and convert to NumPy
    with Timer("Tokenizing"):
        tokenized_output = tokenizer(sentences, padding=False, truncation=True, return_tensors='np')
        input_ids_array = tokenized_output["input_ids"]
        attention_mask_array = tokenized_output["attention_mask"]

    with Timer("Sorting"):
        sorted_indices = np.argsort([-len(ids) for ids in input_ids_array], )
        input_ids_array = input_ids_array[sorted_indices]
        attention_mask_array = attention_mask_array[sorted_indices]

    with Timer("Creating dataset"):
        # dataset = list(zip(input_ids_list, attention_mask_list))
        dataset = np.stack((input_ids_array, attention_mask_array), axis=1)

        # dataset = TensorDataset(tokenized_output["input_ids"], tokenized_output["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # Initialize BERT model and set it to evaluation mode
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Loop through DataLoader to get batches
    for i, (batch_input_ids, batch_attention_mask) in enumerate(tqdm.tqdm(dataloader)):
        #print(f"Batch {i + 1} size {batch_input_ids.shape}")
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        #Tensor of all embeddings
        all_embeddings = []
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            embedding = outputs.last_hidden_state
            all_embeddings.append(embedding)
    return all_embeddings


def unsorted_dyn_batch(sentences, batch_size):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Tokenize sentences at once and convert to NumPy
    tokenized_output = tokenizer(sentences, padding=False, truncation=True, return_tensors='np')
    input_ids_array = tokenized_output["input_ids"]
    attention_mask_array = tokenized_output["attention_mask"]

    # dataset = list(zip(input_ids_list, attention_mask_list))
    dataset = np.stack((input_ids_array, attention_mask_array), axis=1)

    # dataset = TensorDataset(tokenized_output["input_ids"], tokenized_output["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # Initialize BERT model and set it to evaluation mode
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Loop through DataLoader to get batches
    for i, (batch_input_ids, batch_attention_mask) in enumerate(tqdm.tqdm(dataloader)):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            embeddings = outputs.last_hidden_state
            #print(f"Batch {i + 1} embeddings shape: {embeddings.shape}")



if __name__ == "__main__":
    current_time = time.time()
    sentences = initialize_movie_reviews()
    #sentences.extend(sentences)
    #sentences.extend(sentences)
    #sentences = sentences[:1000]
    batch_size = 64
    print(f"Number of sentences: {len(sentences)}")
    last_time = current_time
    current_time = time.time()
    print(f"Initializing took {current_time - last_time} seconds.")
    a = sorted_dyn_batch(sentences, batch_size)
    last_time = current_time
    current_time = time.time()
    print(f"Sorted dynamic batching embedding took {current_time - last_time} seconds.")
    unsorted_dyn_batch(sentences, batch_size)
    last_time = current_time
    current_time = time.time()
    print(f"Unsorted dynamic batching embedding took {current_time - last_time} seconds.")
    normal(sentences, batch_size)
    last_time = current_time
    current_time = time.time()
    print(f"Normal embedding took {current_time - last_time} seconds.")
