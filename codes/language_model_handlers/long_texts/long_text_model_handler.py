from typing import Optional
from transformers import AutoModel, PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding
from datasets import load_dataset
from torch import Tensor
from torch import cat, stack

from torch.utils.data import Dataset, RandomSampler, DataLoader

class TokenizedDataset(Dataset):
    """Dataset for tokens with optional labels."""

    def __init__(self, tokens: BatchEncoding, labels: Optional[list] = None):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        if self.labels:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]

model_name = 'bert-base-uncased'

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = load_dataset('imdb')

long_review = df['test']['text'][21132]

print(f'The review has {len(long_review.split())} words')

# long_review = 'the mand went to the store and bought a gallon of milk'

tokens = tokenizer(long_review, add_special_tokens=False, truncation=False, return_tensors='pt')

# # try:
# #     prediction = model(**tokens)
# # except RuntimeError as e:
# #     print(e)


def split_tokens_into_smaller_chunks(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_lenght: int) -> list[Tensor]:
    result = [tensor[i: i+chunk_size] for i in range(0, len(tensor), stride)]

    if len(result) > 1:
        # ignore chunks with less than minimal_lenght number of tokens
        result = [x for x in result if len(x) >= minimal_chunk_lenght]
    
    return result

def add_special_tokens_at_beggining_and_end(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]):
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        mask_chunks[i] = cat([Tensor([1]), input_id_chunks[i], Tensor([1])])

def add_padding_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = cat([mask_chunks[i], Tensor([0] * pad_len)])

def stack_tokens_from_all_chunks(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Reshapes data to a form compatible with BERT model input."""
    input_ids = stack(input_id_chunks)
    attention_mask = stack(mask_chunks)

    return input_ids.long(), attention_mask.int()

def tokenize_text_with_truncation(text, tokenizer, maximum_text_lenght):
    return tokenizer(text, model_max_length=maximum_text_lenght, add_special_tokens=False, truncation=True, return_tensors='pt')

def tokenize_whole_text(text, tokenizer):
    return tokenizer(text, add_special_tokens=False, truncation=False, return_tensors='pt')

def transform_single_text(text:str, tokenizer:PreTrainedTokenizerBase, chunk_size:int, stride:int, minimal_chunk_lenght:int, maximal_text_lenght: Optional[int]) -> tuple[Tensor, Tensor]:
    if maximal_text_lenght:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_lenght)
    else:
        tokens = tokenize_whole_text(text, tokenizer)
    
    input_id_chunks = split_tokens_into_smaller_chunks(tokens['input_ids'][0], chunk_size, stride, minimal_chunk_lenght)
    mask_chunks = split_tokens_into_smaller_chunks(tokens['attention_mask'][0], chunk_size, stride, minimal_chunk_lenght)

    add_special_tokens_at_beggining_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)

    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)

    return input_ids, attention_mask


def transform_list_of_texts(texts: list[str], tokenizer: PreTrainedTokenizerBase, chunk_size: int, stride: int, minimal_chunk_length: int,
                            maximal_text_length: Optional[int] = None) -> BatchEncoding:
    model_inputs = [transform_single_text(text, tokenizer, chunk_size, stride, minimal_chunk_length, maximal_text_length) for text in texts]
    
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    
    return BatchEncoding(tokens)

def collate_fn_pooled_tokens(data):
    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]
    if len(data[0]) == 2:
        collated = [input_ids, attention_mask]
    else:
        labels = Tensor([data[i][2] for i in range(len(data))])
        collated = [input_ids, attention_mask, labels]
    return collated

# ------------------------------------------------------------------------------------------------------------------------------

# example_tensor = tokens['input_ids'][0]
# print(example_tensor)

# splitted = split_tokens_into_smaller_chunks(tensor=example_tensor, chunk_size=5, stride=3, minimal_chunk_lenght=5)

# print(splitted)

# input_ids, attention_mask = transform_single_text(text=long_review, tokenizer=tokenizer, 
#                                                 chunk_size=510, stride=510, minimal_chunk_lenght=1,
#                                                 maximal_text_lenght=None)

# print(input_ids)
# print(attention_mask)

short_review = df["test"]["text"][0]
number_of_words = len(short_review.split())
print(f"The review has {number_of_words} words.")

def tokenize_truncated(list_of_texts):
    return tokenizer(list_of_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

tokens_splitted = transform_list_of_texts([short_review, long_review], tokenizer, 510, 510, 1, None)
tokens_truncated = tokenize_truncated([short_review, long_review])

print(type(tokens_truncated["input_ids"]))

print(tokens_truncated['input_ids'].shape)

print(type(tokens_splitted["input_ids"]))

print([tensor.shape for tensor in tokens_splitted['input_ids']])


dataset_truncated = TokenizedDataset(tokens_truncated, [0,1])
dataset_splitted = TokenizedDataset(tokens_splitted, [0,1])
train_dataloader_truncated = DataLoader(dataset_truncated, sampler=RandomSampler(dataset_truncated), batch_size=2)
train_dataloader_splitted = DataLoader(dataset_splitted, sampler=RandomSampler(dataset_splitted), batch_size=2)

train_dataloader_splitted = DataLoader(dataset_splitted, sampler=RandomSampler(dataset_splitted), batch_size=2, collate_fn=collate_fn_pooled_tokens)
