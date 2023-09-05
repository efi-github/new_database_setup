import copy
import sys
import time
from pprint import pprint

#from cuml import UMAP as cumlUMAP
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List, Any
import torch
from fastapi import Depends
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from pydantic import BaseModel, Field
import numpy as np
import umap
import tqdm
from utilities import Timer

from db.session import get_db


class Model(BaseModel):
    def fit(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")

    def transform(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")
"""
class C_Umap(Model):
    umap_arguments: dict = Field(dict(), description="Arguments for Umap")
    name: str = ""
    fitted: bool = False

    def fit(self, data: Union[np.ndarray, list]) -> bool:
        if len(data) == 0:
            raise ValueError("The data is empty.")
        self._model = umap.UMAP(**self.umap_arguments)
        self._model.fit(data)
        self.fitted = True
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
        return f"C_Umap({self.umap_arguments})"
        
"""

class Umap(Model):
    arguments: dict = Field(dict(), description="Arguments for Umap")
    name: str = ""
    fitted: bool = False

    def fit(self, data: Union[np.ndarray, list]) -> bool:
        if len(data) == 0:
            raise ValueError("The data is empty.")
        self._model = umap.UMAP(**self.arguments)
        self._model.fit(data)
        self.fitted = True
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
        return f"Umap({self.arguments})"


class SemiSupervisedUmap(Umap):

    def fit(self, data: Union[np.ndarray, list], labels: np.ndarray = None) -> bool:
        print(f"SemiSupervisedUmap.fit() from {self.arguments}")
        if len(data) == 0:
            raise ValueError("The data is empty.")
        self._model = umap.UMAP(**self.arguments)
        self._model.fit(data, y=labels)
        self.fitted = True
        return True

    def __str__(self):
        return f"SemiSupervisedUmap({self.arguments})"


class BertEmbeddingModel(BaseModel):
    arguments: dict = Field({"pretrained_model_name_or_path": "bert-base-uncased"},
                                 description="Arguments for BERT")
    name: str = ""
    fitted: bool = False
    default_parameters: dict = {"pretrained_model_name_or_path": "bert-base-uncased"}

    def fit(self, segments, sentences):
        print(f"BertEmbedding.fit() from {self.arguments}")
        if segments is None or sentences is None:
            raise ValueError("The data is empty.")
        self.fitted = True

    def collate_fn(self, batch):
        input_ids, attention_mask, offset_mapping, sentence_id = zip(*batch)
        input_ids = [torch.tensor(x) for x in input_ids]
        attention_mask = [torch.tensor(x) for x in attention_mask]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return input_ids, attention_mask, offset_mapping, sentence_id

    def transform(self, segments, sentences):
        unique_sentences = list(set(sentences))
        print(f"BertEmbedding.transform() with #{len(segments)} segments")
        if len(segments) == 0:
            return np.array([])
        with Timer("Calculate Sentence Embeddings"):
            sentence_embedding = self.transform_sentences(unique_sentences)
        with Timer("Calculating Segment Offset Indexes"):
            positions = self.get_segment_positions(segments, sentences, sentence_embedding)
        return self.segment_embedding([sentence_embedding[s.SentenceID][0] for s in sentences], positions)

    def transform_sentences(self, sentences):
        temp = copy.deepcopy(self.default_parameters)
        temp.update(self.arguments)
        self.arguments = temp
        tokenizer = BertTokenizerFast.from_pretrained(**self.arguments)
        model = BertModel.from_pretrained(**self.arguments)
        max_input_length = model.config.max_position_embeddings

        if torch.cuda.is_available():
            model.to('cuda')
            torch.cuda.empty_cache()

        # Das speichern der Satz ids und texte.
        with Timer("Fetching Dataset"):
            sentences_id = np.array([s.SentenceID for s in sentences])
            sentences_text = [s.Text for s in sentences]

        # Tokenization
        with Timer("Tokenization"):
            inputs = tokenizer(sentences_text, return_tensors="np", return_offsets_mapping=True,
                               return_attention_mask=True, max_length=max_input_length)

        # Sortieren der Sätze nach der Länge
        with Timer("Sorting"):
            # sortiert von lang nach kurz, damit es schnell abbricht falls die maximale größe nicht unterstützt wird
            sorted_indices = np.argsort([-len(ids) for ids in inputs['input_ids']])
            sorted_input_ids_array = inputs['input_ids'][sorted_indices]
            sorted_attention_mask_array = inputs['attention_mask'][sorted_indices]
            sorted_offset_mapping_array = inputs['offset_mapping'][sorted_indices]
            sorted_sentences_id = sentences_id[sorted_indices]

        # Erstellen des Datasets und Dataloader
        with Timer("Creating Dataset and Loader"):
            BATCH_SIZE = 124
            dataset = np.stack(
                (sorted_input_ids_array, sorted_attention_mask_array, sorted_offset_mapping_array, sorted_sentences_id),
                axis=1)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=self.collate_fn)

        model.eval()  # Set the model to evaluation mode

        all_embeddings = {}

        for i, (batch_input_ids, batch_attention_mask, batch_offset_mapping, batch_sentence_ids) in enumerate(
                tqdm.tqdm(dataloader)):

            # Alle 10 batches die GPU leeren
            if i % 10 == 0:
                torch.cuda.empty_cache()

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
                embeddings = embeddings.cpu().numpy()
                all_embeddings.update({id: (embedding, offset, mask) for id, embedding, offset, mask in
                                       zip(batch_sentence_ids, embeddings, batch_offset_mapping, batch_attention_mask)})

        return all_embeddings

    def get_segment_positions(self, segments, sentence, embeddings):
        def find_subrange(true_range, all_ranges):
            start, end = true_range
            mask = (all_ranges[:, 0] <= end) & (all_ranges[:, 1] >= start)
            mask = torch.Tensor(mask)
            result = torch.nonzero(mask).squeeze().tolist()
            del mask
            return result

        results = []
        for i in range(len(segments)):
            id = sentence[i].SentenceID
            start = segments[i].StartPosition
            len_seg = len(segments[i].Text)
            attention_mask = embeddings[id][2]
            offset = embeddings[id][1]
            valid_length = torch.where(attention_mask == 0)[0][0] if 0 in attention_mask else len(
                attention_mask)
            temp_offsets = offset[:valid_length]
            result = find_subrange((start, start + len_seg), temp_offsets)
            if type(result) != type([]):
                result = [result]
            results.append(result)
            del valid_length, temp_offsets, start, len_seg

        return results

    def segment_embedding(self, embeddings, positions):
        averaged_embeddings = []
        for emb, pos in zip(embeddings, positions):
            selected_embeddings = emb[pos, :]
            mean_embedding = np.mean(selected_embeddings, axis=0)
            averaged_embeddings.append(mean_embedding)
        averaged_embeddings = np.array(averaged_embeddings)
        return averaged_embeddings

    def get_segment_string(self, segment):
        sentence = segment.sentence.Text
        return sentence

    def __str__(self):
        temp = copy.deepcopy(self.default_parameters)
        temp.update(self.arguments)
        return f"BertEmbeddingModel({temp})"


MODELS = {
    "umap": Umap,
    "semisupervised_umap": SemiSupervisedUmap,
    "bert": BertEmbeddingModel,
#    "cuml_umap": C_Umap
}


def test():
    pass
