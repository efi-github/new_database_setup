from pprint import pprint

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text = "Hello, how are you?"

encoding = tokenizer.encode_plus(
    text,
    return_offsets_mapping=True,
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    max_length=20,
    return_attention_mask=True,
    return_tensors="pt"  # Return PyTorch tensors
)

pprint(encoding)