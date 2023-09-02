from typing import Union, List, Any
import torch
from transformers import BertTokenizer, BertModel
from pydantic import BaseModel, Field
import numpy as np
import umap



class Model(BaseModel):
    def fit(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")

    def transform(self, data):
        raise NotImplementedError("This method should be implemented in a child class.")



class Umap(Model):
    umap_arguments: dict = Field(dict(), description="Arguments for Umap")

    def fit(self, data: Union[np.ndarray, list]) -> bool:
        self._model = umap.UMAP(**self.umap_arguments)
        self._model.fit(data)
        return True

    def transform(self, data: Union[np.ndarray, list]) -> np.ndarray:
        if self._model is None:
            raise ValueError("The UMAP model has not been fitted yet.")
        transformed_data = self._model.transform(data)
        return transformed_data

    def __str__(self):
        return f"Umap({self.umap_arguments})"


class SemiSupervisedUmap(Umap):

    def fit(self, data: Union[np.ndarray, list], labels: np.ndarray = None) -> bool:
        self._model = umap.UMAP(**self.umap_arguments)
        self._model.fit(data, y=labels)
        return True

    def __str__(self):
        return f"SemiSupervisedUmap({self.umap_arguments})"




class BertEmbeddingModel(BaseModel):
    bert_arguments: dict = Field({"pretrained_model_name_or_path" : "bert-base-uncased"}, description="Arguments for BERT")
    #huggingface_model_name: str = Field("bert-base-uncased", description="Name of the BERT model to be used.")
    tokenizer: Any = None
    model: Any = None

    def fit(self):
        print(self.bert_arguments)
        # Load the tokenizer and model from Hugging Face
        self.tokenizer = BertTokenizer.from_pretrained(**self.bert_arguments)
        self.model = BertModel.from_pretrained(**self.bert_arguments)

        # Move the model to the GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

    def transform(self, sentences) -> torch.Tensor:

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        # Move the inputs to the GPU if available
        if torch.cuda.is_available():
            inputs = {key: tensor.to('cuda') for key, tensor in inputs.items()}

        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        return self.segment_embedding(embeddings.cpu().numpy())

    def segment_embedding(self, embeddings):
        return np.mean(embeddings, axis=1)

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