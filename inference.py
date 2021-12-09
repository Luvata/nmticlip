import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm, trange
import argparse
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


D = torch.device
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                #                 layers.append(nn.Dropout(p=0.1))
                layers.append(nn.Dropout(p=0.2))  # update 17/10 20:00PM

        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    #     stop_token_index = 0

    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def parse_model_str(gspath):
    model_str = gspath.split("/")[-1]
    loss, model_type, trainds, valds, seed_and_epoch = model_str.split("_")
    loss = float(loss)
    epoch = int(seed_and_epoch.split("-")[1][:3])
    seed = int(seed_and_epoch.split("-")[0][-1])
    return {
        "model_str": model_str,
        "loss": loss,
        "model_type": model_type,
        "train": trainds,
        "val": valds,
        "seed": seed,
        "epoch": epoch,
    }


def get_model_type_from_config(config):
    if config["model_type"] == "mlp":
        return ClipCaptionPrefix(10)
    elif config["model_type"] == "mlpgpt":
        return ClipCaptionModel(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-str", type=str, required=True, help="Path to pretrained model in GS"
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=4,
        help="Number of CPU threads for torch inference, default 4",
    )
    args = parser.parse_args()
    print(">>>>>>>>", args)
    prefix_length = 10 # TODO hardcoded

    torch.set_num_threads(args.nthread)

    model_str = args.model_str
    config = parse_model_str(model_str)

    # download model from gs
    os.system(f"gsutil cp {model_str} .")
    model_path = model_str.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
    model = get_model_type_from_config(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    from datasets import load_metric

    metric = load_metric("sacrebleu")

    if config["val"] == "viecap":
        data = torch.load("./viecap_clean/test_viecap_1k.pt")
        embedding, refs = data["clip_embedding"], data["target_sentence"]
    elif config["val"] == "sat":
        data = torch.load("./viecap_clean/test_sat_1k.pt")
        embedding, refs = data["clip_embedding"], [
            [tokenizer.decode(sent, skip_special_tokens=True)] for sent in data["target"]
        ]

    device = "cpu"
    with torch.no_grad():
        all_embed = model.clip_project(embedding.float().to(device))
        all_embed = all_embed.view(all_embed.shape[0], 1, prefix_length, -1)
        print(all_embed.shape)

    predictions = [generate_beam(model, tokenizer, embed=i)[0].split("\n")[0] for i in tqdm(all_embed)]
    # write predictions to file with name of model_path and score
    with open(f"{model_path}_predictions.txt", "w") as f:
        for pred in predictions:
            f.write(pred + "\n")
    score = metric.compute(predictions=predictions, references=refs)
    print(score)


if __name__ == "__main__":
    main()
