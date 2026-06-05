"""Data Pipeline Module."""
import re
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class WordTokenizer:
    """Tokenizador simple que separa estrictamente por palabras
    completas y signos de puntuación."""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def fit_on_text(self, text_lines: List[str]):
        """Construye el vocabulario usando solo palabras del corpus."""
        idx = len(self.vocab)
        for line in text_lines:
            words = re.findall(r"\w+|[^\w\s]", line.lower(), re.UNICODE)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    self.inverse_vocab[idx] = word
                    idx += 1
        self.vocab_size = len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizer."""
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encoder."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decoder."""
        tokens = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            token = self.inverse_vocab.get(i, "[UNK]")
            if skip_special_tokens and token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [
            self.inverse_vocab.get(
                i.item()
                if isinstance(i, torch.Tensor)
                else i, "[UNK]") for i in ids
                ]

    @property
    def pad_token_id(self):
        """Pad Token Id."""
        return self.vocab["[PAD]"]

    @property
    def cls_token_id(self):
        """Token Id."""
        return self.vocab["[CLS]"]

    @property
    def sep_token_id(self):
        """Sep Token Id."""
        return self.vocab["[SEP]"]

    @property
    def mask_token_id(self):
        """Mask Token Id."""
        return self.vocab["[MASK]"]

    @property
    def unk_token_id(self):
        """UNK Token Id."""
        return self.vocab["[UNK]"]


class BERTTextDataset(Dataset):
    """Bert Text Dataset."""
    def __init__(
            self,
            txt_file_path: str,
            tokenizer: WordTokenizer,
            max_len: int = 128
            ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self._read_and_pair_text(txt_file_path)

    def _read_and_pair_text(self, filepath: str) -> List[Tuple[str, str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        pairs = []
        for i in range(len(lines) - 1):
            pairs.append((lines[i], lines[i+1]))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text_a, text_b = self.pairs[idx]
        tokens_a = self.tokenizer.encode(text_a)
        tokens_b = self.tokenizer.encode(text_b)

        avail_space = self.max_len - 3
        while len(tokens_a) + len(tokens_b) > avail_space:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        input_ids = [self.tokenizer.cls_token_id] + tokens_a + \
                    [self.tokenizer.sep_token_id] + tokens_b +  \
                    [self.tokenizer.sep_token_id]
        segment_ids = [1] * (len(tokens_a) + 2) + [2] * (len(tokens_b) + 1)

        padding_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        segment_ids += [0] * padding_len

        return torch.tensor(
            input_ids, dtype=torch.long
            ), torch.tensor(segment_ids, dtype=torch.long)


class BERTCollateFn:
    """BERTCollateFn."""
    def __init__(self, tokenizer: WordTokenizer):
        self.tokenizer = tokenizer

    def __call__(
            self,
            batch: List[Tuple[torch.Tensor, torch.Tensor]]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        segment_ids = torch.stack([item[1] for item in batch])
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.mask_token_id
        ]

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sp_id in special_tokens:
            special_tokens_mask |= (input_ids == sp_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)
            ).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)
            ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            5,
            self.tokenizer.vocab_size,
            labels.shape,
            dtype=torch.long
            )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, segment_ids, labels


class BERTTextDataModule(pl.LightningDataModule):
    """Bert Text Data Module."""
    def __init__(
            self,
            txt_file_path: str,
            batch_size: int,
            max_len: int,
            num_workers: int = 2,
            val_split: float = 0.2
            ):
        super().__init__()
        self.txt_file_path = txt_file_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.val_split = val_split
        self.tokenizer = WordTokenizer()

    def setup(self, stage: Optional[str] = None):
        with open(self.txt_file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.tokenizer.fit_on_text(lines)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        full_dataset = BERTTextDataset(self.txt_file_path, self.tokenizer, self.max_len)

        total_len = len(full_dataset)
        val_len = max(1, int(total_len * self.val_split))
        train_len = total_len - val_len

        if total_len < 2:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, val_len]
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=BERTCollateFn(self.tokenizer),
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=BERTCollateFn(self.tokenizer),
            pin_memory=True
        )
