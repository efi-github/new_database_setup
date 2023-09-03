import time
from bisect import bisect_left

from typing import Union, List, Any
import torch
from fastapi import Depends
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from pydantic import BaseModel, Field
import numpy as np
import umap
import tqdm

from db.session import get_db


class Model(BaseModel):
    def fit(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")

    def transform(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")



class Umap(Model):
    umap_arguments: dict = Field(dict(), description="Arguments for Umap")
    name: str = ""

    def fit(self, data: Union[np.ndarray, list]) -> bool:
        if len(data) == 0:
            raise ValueError("The data is empty.")
        self._model = umap.UMAP(**self.umap_arguments)
        self._model.fit(data)
        return True

    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        if len(data) == 0:
            return np.array([])
        print(f"Umap.transform() with #{len(data)} embeddings")
        if self._model is None:
            raise ValueError("The UMAP model has not been fitted yet.")
        transformed_data = self._model.transform(data)
        return transformed_data

    def __str__(self):
        return f"Umap({self.umap_arguments})"


class SemiSupervisedUmap(Umap):

    def fit(self, data: Union[np.ndarray, list], labels: np.ndarray = None) -> bool:
        print(f"SemiSupervisedUmap.fit() from {self.umap_arguments}")
        if len(data) == 0:
            raise ValueError("The data is empty.")
        self._model = umap.UMAP(**self.umap_arguments)
        self._model.fit(data, y=labels)
        return True

    def __str__(self):
        return f"SemiSupervisedUmap({self.umap_arguments})"




class BertEmbeddingModel(BaseModel):
    bert_arguments: dict = Field({"pretrained_model_name_or_path" : "bert-base-uncased"}, description="Arguments for BERT")
    name: str = ""

    def fit(self, segments):
        print(f"BertEmbedding.fit() from {self.bert_arguments}")
        if len(segments) == 0:
            raise ValueError("The data is empty.")


    def transform_sentences(self, sentences):
        tokenizer = BertTokenizerFast.from_pretrained(**self.bert_arguments)
        model = BertModel.from_pretrained(**self.bert_arguments)
        if torch.cuda.is_available():
            model.to('cuda')

        sentences_id = [s.SentenceID for s in sentences]
        sentences_text = [s.Text for s in sentences]
        inputs = tokenizer(sentences_text, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True,
                           return_attention_mask=True)

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        BATCH_SIZE = 256
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()  # Set the model to evaluation mode

        all_embeddings = {}

        for i, (batch_input_ids, batch_attention_mask) in enumerate(tqdm.tqdm(dataloader)):
            # Prepare the batch
            batch_inputs = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }

            if torch.cuda.is_available():
                batch_inputs = {key: tensor.to('cuda') for key, tensor in batch_inputs.items()}

            with torch.no_grad():
                outputs = model(**batch_inputs)
                embeddings = outputs.last_hidden_state
                batch_ids = sentences_id[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                all_embeddings.update({id: embedding for id, embedding in zip(batch_ids, embeddings)})
        return all_embeddings

    def transform(self, segments_and_sentences) -> torch.Tensor:
        segments, sentences = segments_and_sentences[0], segments_and_sentences[1]
        unique_sentences = list(set(sentences))
        print(f"BertEmbedding.transform() with #{len(segments)} segments")
        if len(segments) == 0:
            return np.array([])
        sentence_embedding = self.transform_sentences(unique_sentences)
        origin_time = time.time()
        current_time = origin_time
        # Load the tokenizer and model from Hugging Face
        tokenizer = BertTokenizerFast.from_pretrained(**self.bert_arguments)
        model = BertModel.from_pretrained(**self.bert_arguments)

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
        last_time = current_time
        current_time = time.time()
        print(f"Loading the model took {current_time - last_time} seconds.")
        sentences = [s.sentence.Text for s in segments]
        last_time = current_time
        current_time = time.time()
        print(f"fetching sentences took {current_time - last_time} seconds.")
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True, return_attention_mask=True)
        del sentences
        last_time = current_time
        current_time = time.time()
        print(f"Tokenization took {current_time - last_time} seconds.")

        positions = self.get_segment_positions(inputs, segments)

        last_time = current_time
        current_time = time.time()
        print(f"Getting Positions of the segments took {current_time - last_time} seconds.")

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        BATCH_SIZE = 256
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # Use your desired batch size

        last_time = current_time
        current_time = time.time()
        print(f"Setting up set and loader took {current_time - last_time} seconds.")

        model.eval()  # Set the model to evaluation mode

        all_embeddings = []

        for i, (batch_input_ids, batch_attention_mask) in enumerate(tqdm.tqdm(dataloader)):
            # Prepare the batch
            batch_inputs = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }
            batch_positions = positions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            if torch.cuda.is_available():
                batch_inputs = {key: tensor.to('cuda') for key, tensor in batch_inputs.items()}

            with torch.no_grad():
                outputs = model(**batch_inputs)
                embeddings = outputs.last_hidden_state

                all_embeddings.append(torch.tensor(self.segment_embedding(embeddings.cpu().numpy(), batch_positions)))

        # Combine all embeddings
        del batch_input_ids, batch_attention_mask, batch_inputs, outputs
        torch.cuda.empty_cache()
        final_embeddings = torch.cat(all_embeddings, dim=0)

        # Move final tensor to CPU and convert to NumPy
        #final_embeddings_numpy = final_embeddings.cpu().numpy()

        print(f"Generating embeddings took {time.time() - current_time} seconds.")
        print(f"Total time taken: {time.time() - origin_time} seconds.")
        return final_embeddings


    def transform1(self, segments) -> torch.Tensor:
        print(f"BertEmbedding.transform() with #{len(segments)} segments")
        if len(segments) == 0:
            return np.array([])
        origin_time = time.time()
        current_time = origin_time
        # Load the tokenizer and model from Hugging Face
        tokenizer = BertTokenizerFast.from_pretrained(**self.bert_arguments)
        model = BertModel.from_pretrained(**self.bert_arguments)

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
        last_time = current_time
        current_time = time.time()
        print(f"Loading the model took {current_time - last_time} seconds.")
        strings = list(map(self.get_segment_string, segments))
        inputs = tokenizer(strings, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True, return_attention_mask=True)

        positions = list(map(self.get_segment_position, [{key: [value] for key, value in zip(inputs.keys(), values)}
                                                                for values in zip(*inputs.values())
                                                        ], segments))
        last_time = current_time
        current_time = time.time()
        print(f"Getting Positions of the segments took {current_time - last_time} seconds.")
        strings = list(map(self.get_segment_string, segments))
        inputs = tokenizer(strings, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True,
                           return_attention_mask=True)
        last_time = current_time
        current_time = time.time()
        print(f"Tokenizing the segments took {current_time - last_time} seconds.")
        batch_size = 32  # Choose an appropriate batch size
        num_batches = int(np.ceil(len(segments) / batch_size))

        all_embeddings = []

        for i in tqdm.tqdm(range(num_batches)):
            # Extract the batch from the inputs
            batch_input_ids = inputs['input_ids'][i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = inputs['attention_mask'][i * batch_size: (i + 1) * batch_size]
            positions_batch = positions[i * batch_size: (i + 1) * batch_size]

            # Prepare the input dictionary for this batch
            batch_inputs = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }

            # Move the batch to the GPU if available
            if torch.cuda.is_available():
                batch_inputs = {key: tensor.to('cuda') for key, tensor in batch_inputs.items()}

            # Generate embeddings for this batch
            # Generate embeddings for this batch
            with torch.no_grad():
                outputs = model(**batch_inputs)
                embeddings = outputs.last_hidden_state

                # Append the tensor to the list without moving it to CPU or converting to NumPy
                all_embeddings.append(self.segment_embedding(embeddings.cpu(), positions_batch))


        print(f"Generating embeddings took {time.time() - current_time} seconds.")
        print(f"Total time taken: {time.time() - origin_time} seconds.")
        return all_embeddings#self.segment_embedding(final_embeddings.cpu().numpy(), positions)



    def segment_embedding(self, embeddings, positions):
        averaged_embeddings = []
        print(positions)
        for emb, pos in zip(embeddings, positions):
            if len(pos) == 0:
                print(emb)
            selected_embeddings = emb[pos, :]
            mean_embedding = np.mean(selected_embeddings, axis=0)
            averaged_embeddings.append(mean_embedding)
        averaged_embeddings = np.array(averaged_embeddings)
        return averaged_embeddings

    def get_segment_string(self, segment, db = Depends(get_db)):
        sentence = segment.sentence.Text
        segment = segment.Text
        return sentence

    def get_segment_positions(self, tokenization, segments):
        def find_subrange(true_range, all_ranges):
            start, end = true_range
            mask = (all_ranges[:, 0] <= end) & (all_ranges[:, 1] >= start)
            result = torch.nonzero(mask).squeeze().tolist()
            del mask
            return result

        # Assuming that tokenization['attention_mask'] and tokenization['offset_mapping'] are already torch tensors
        attention_masks = tokenization['attention_mask']
        offset_mappings = tokenization['offset_mapping']


        results = []
        for i in range(len(segments)):
            start = segments[i].StartPosition
            len_seg = len(segments[i].Text)
            valid_length = torch.where(attention_masks[i, :] == 0)[0][0] if 0 in attention_masks[i, :] else len(attention_masks[i, :])
            temp_offsets = offset_mappings[i, :][:valid_length]
            result = find_subrange((start, start + len_seg), temp_offsets)
            if type(result) != type([]):
                result = [result]
            results.append(result)
            del valid_length, temp_offsets, start, len_seg

        return results


    def __str__(self):
        return f"BertEmbeddingModel({self.bert_arguments})"


MODELS = {
    "umap": Umap,
    "semisupervised_umap": SemiSupervisedUmap,
    "bert": BertEmbeddingModel
}


def test():
    bert_model = BertEmbeddingModel(bert_arguments = {"pretrained_model_name_or_path" : "dslim/bert-base-NER"})
    bert_model.fit()
    embeddings = bert_model.transform(sentences=[f"This is entence nr.{i}" for i in range(100)])
    print(bert_model)
    print(embeddings.shape)
    print(embeddings[:5])

    umap_model = Umap()
    umap_model.fit(embeddings)
    reduced_embeddings = umap_model.transform(embeddings)
    print(umap_model)
    print(reduced_embeddings.shape)
    print(reduced_embeddings[:5])
    print(reduced_embeddings[-5:])



"""    def transform1(self, segments) -> torch.Tensor:
        print(f"BertEmbedding.transform() with #{len(segments)} segments")
        if len(segments) == 0:
            return np.array([])
        # Load the tokenizer and model from Hugging Face
        tokenizer = BertTokenizerFast.from_pretrained(**self.bert_arguments)
        model = BertModel.from_pretrained(**self.bert_arguments)

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            model.to('cuda')

        strings = list(map(self.get_segment_string, segments))
        inputs = tokenizer(strings, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True, return_attention_mask=True)

        positions = list(map(self.get_segment_position, [{key: [value] for key, value in zip(inputs.keys(), values)}
                                                                for values in zip(*inputs.values())
                                                        ], segments))
        # Move the inputs to the GPU if available
        if torch.cuda.is_available():
            inputs = {key: tensor.to('cuda') for key, tensor in inputs.items() if key in ['input_ids', 'attention_mask']}

        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

        return self.segment_embedding(embeddings.cpu().numpy(), positions)"""