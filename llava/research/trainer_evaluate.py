# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    eval_start_index: int = field(default=1, metadata={"help": "Start index for evaluation samples"})
    eval_end_index: int = field(default=100, metadata={"help": "End index for evaluation samples"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    labels = input_ids
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    labels_lens = input_ids_lens
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    

    #------------------------------------------------------------—--------
    from torch.utils.data import Subset, DataLoader
    trainer.model.eval()
    NUM_IMAGE_TOKENS=int(os.environ.get("NUM_IMAGE_TOKENS"))
    PRUNE_LAYER_INDEX=int(os.environ.get("PRUNE_LAYER_INDEX"))

    # 循环处理样本
    for sample_image_index in range(training_args.eval_start_index, training_args.eval_end_index + 1):
        print(f"\n{'='*60}")
        print(f"=== Processing sample {sample_image_index} ===")
        print(f"{'='*60}")
        
        sample_indices = [sample_image_index]
        one_ds = Subset(data_module["train_dataset"], sample_indices)
        eval_loader = DataLoader(one_ds, batch_size=1, collate_fn=trainer.data_collator)
        sample_batch = next(iter(eval_loader))
        labels = sample_batch.get('labels')
        input_ids = sample_batch['input_ids']
        
        # 获取原始图像路径
        sample_data = data_module["train_dataset"].list_data_dict[sample_indices[0]]
        image_file = sample_data['image']
        image_folder = data_args.image_folder
        original_image_path = os.path.join(image_folder, image_file)
        original_image = Image.open(original_image_path).convert('RGB')
        print(f"[DEBUG] Original image loaded from: {original_image_path}")
        print(f"[DEBUG] Original image size: {original_image.size}")
        
        if labels is not None:
            valid_labels = (labels != IGNORE_INDEX).sum().item()
            total_labels = labels.numel()
            print(f"[DEBUG] Labels: valid={valid_labels}, total={total_labels}, ignore={total_labels - valid_labels}")
        print(f"[DEBUG] Number of image placeholders in prompt: {NUM_IMAGE_TOKENS}")

        # ================================================================
        # 1) BASELINE: 完整forward，不剪枝任何视觉token
        # ================================================================
        print(f"\n{'='*60}")
        print(f"=== BASELINE: Full forward (no pruning) ===")
        print(f"{'='*60}")
        
        os.environ["PRUNE_TOKEN_INDEX"] = "-1"  # 清除单独剪枝的token索引
        
        baseline_metrics = trainer.evaluate(eval_dataset=one_ds)
        print(f"[BASELINE] EVAL metrics:", baseline_metrics)
        baseline_loss = baseline_metrics.get("eval_loss", None)
        print(f"[BASELINE] Loss: {baseline_loss}")
        
        # 获取baseline的predictions (logits)
        baseline_pred = trainer.predict(test_dataset=one_ds)
        baseline_logits = baseline_pred.predictions  # (seq_len, vocab_size)
        print(f"[BASELINE] logits shape: {baseline_logits.shape}")

        # ================================================================
        # 2) 循环剪枝每一个视觉token，计算loss差值
        # ================================================================
        print(f"\n{'='*60}")
        print(f"=== PRUNE EACH TOKEN: Loop through each visual token ===")
        print(f"{'='*60}")
        
        print(f"Will prune tokens from index 0 to {NUM_IMAGE_TOKENS - 1}")
        print(f"Number of image placeholders in prompt: {NUM_IMAGE_TOKENS}")
        
        loss_differences = []
        all_losses = []
        
        for prune_idx in range(NUM_IMAGE_TOKENS):
            if prune_idx % 10 == 0:
                print(f"\n--- Pruning token index: {prune_idx} ---")
            
            # 设置剪枝模式
            os.environ["PRUNE_TOKEN_INDEX"] = str(prune_idx)  # 设置要剪枝的token索引
            
            # 计算loss
            metrics = trainer.evaluate(eval_dataset=one_ds)
            current_loss = metrics.get("eval_loss", None)
            
            if current_loss is not None and baseline_loss is not None:
                loss_diff = current_loss - baseline_loss
                loss_differences.append(loss_diff)
                all_losses.append(current_loss)
            else:
                loss_differences.append(0.0)
                all_losses.append(0.0)
            
            if prune_idx % 10 == 0:
                print(f"  Prune token {prune_idx}: loss={current_loss}, diff={loss_diff}")
        
        # 恢复设置
        os.environ["PRUNE_TOKEN_INDEX"] = "-1"
        
        # ================================================================
        # 3) 绘制差值图
        # ================================================================
        print(f"\n{'='*60}")
        print(f"=== PLOTTING: Loss difference vs baseline ===")
        print(f"{'='*60}")
        
        # 定义平滑函数 (滑动平均)
        def smooth_data(data, window_size=10):
            """对数据进行滑动平均平滑"""
            if window_size <= 1:
                return data
            kernel = np.ones(window_size) / window_size
            # 使用卷积，'same' 保持输出长度与输入相同
            smoothed = np.convolve(data, kernel, mode='same')
            return smoothed
        
        # 设置平滑窗口大小
        smooth_window = 20
            
        # 创建输出目录
        output_dir = f"/data/users/Actor/results_lossing_{PRUNE_LAYER_INDEX}/{sample_image_index}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] Output directory created: {output_dir}")
        
        # 平滑处理
        loss_differences_smooth = smooth_data(np.array(loss_differences), smooth_window)
        all_losses_smooth = smooth_data(np.array(all_losses), smooth_window)
        
        # 图1: loss差值图（原始 + 平滑）
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loss_differences)), loss_differences, 'b-', alpha=0.3, linewidth=1, label='Original')
        plt.plot(range(len(loss_differences_smooth)), loss_differences_smooth, 'b-', linewidth=2, label=f'Smoothed (window={smooth_window})')
        plt.xlabel('Pruned Token Index', fontsize=12)
        plt.ylabel('Loss Difference (Pruned - Baseline)', fontsize=12)
        plt.title('Loss Difference vs Baseline for Each Pruned Visual Token', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Baseline')
        plt.legend()
        plt.tight_layout()
        
        output_path_1 = f"{output_dir}/prune_token_loss_difference.png"
        plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
        print(f"Loss difference plot saved to: {output_path_1}")
        plt.close()
        
        # 图2: loss图
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(all_losses)), all_losses, 'g-', linewidth=1.5, label='Loss after pruning')
        plt.axhline(y=baseline_loss, color='r', linestyle='--', linewidth=1, label='Baseline loss')
        plt.xlabel('Pruned Token Index', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss for Each Pruned Visual Token', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_path_2 = f"{output_dir}/prune_token_loss.png"
        plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to: {output_path_2}")
        plt.close()
        
        # 图3: loss差值直方图分布
        plt.figure(figsize=(10, 6))
        plt.hist(loss_differences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
        plt.axvline(x=np.mean(loss_differences), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(loss_differences):.4f}')
        plt.xlabel('Loss Difference', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Loss Differences', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path_3 = f"{output_dir}/prune_token_loss_history.png"
        plt.savefig(output_path_3, dpi=150, bbox_inches='tight')
        print(f"Loss histogram saved to: {output_path_3}")
        plt.close()
        
        # 同时保存数据到文件
        data_output_path = f"{output_dir}/prune_token_loss_data.npz"
        np.savez(data_output_path, 
                loss_differences=loss_differences, 
                all_losses=all_losses,
                baseline_loss=baseline_loss)
        print(f"Data saved to: {data_output_path}")
        
        # ================================================================
        # 4) 收集loss差值由大到小的索引 (键是差值，值是索引)
        # ================================================================
        print(f"\n{'='*60}")
        print(f"=== SORTING: Loss difference from large to small ===")
        print(f"{'='*60}")
        
        # 创建字典：键是差值，值是索引
        # 由于差值可能重复，我们使用列表存储相同差值的多个索引
        loss_diff_dict = {}
        for idx, diff in enumerate(loss_differences):
            if diff not in loss_diff_dict:
                loss_diff_dict[diff] = []
            loss_diff_dict[diff].append(idx)
        
        # 按差值从大到小排序
        sorted_diffs = sorted(loss_diff_dict.keys(), reverse=True)
        
        # 创建有序字典：键是差值，值是索引（如果是多个索引，取第一个或全部）
        loss_diff_sorted = {}
        for diff in sorted_diffs:
            indices = loss_diff_dict[diff]
            # 如果只有一个索引，直接存索引；否则存列表
            loss_diff_sorted[diff] = indices[0] if len(indices) == 1 else indices
        
        # 打印前20个影响最大的token
        print(f"\nTop 20 tokens with largest loss difference (most important):")
        for i, diff in enumerate(sorted_diffs[:20]):
            idx = loss_diff_sorted[diff]
            print(f"  Rank {i+1}: diff={diff:.6f}, token_index={idx}")
        
        # 保存排序后的结果到文件
        sorted_output_path = f"{output_dir}/loss_diff_sorted.json"
        # 将键转换为字符串（因为JSON不支持float作为键）
        loss_diff_sorted_str = {str(k): v for k, v in loss_diff_sorted.items()}
        import json as json_lib
        with open(sorted_output_path, 'w') as f:
            json_lib.dump(loss_diff_sorted_str, f, indent=2)
        print(f"\nSorted loss differences saved to: {sorted_output_path}")
        
        # 打印一些统计信息
        print(f"\n=== Summary Statistics ===")
        print(f"Baseline loss: {baseline_loss:.6f}")
        print(f"Max loss difference: {max(loss_differences):.6f}")
        print(f"Min loss difference: {min(loss_differences):.6f}")
        print(f"Mean loss difference: {np.mean(loss_differences):.6f}")
        
        # 找到影响最大和最小的token
        max_diff_idx = np.argmax(loss_differences)
        min_diff_idx = np.argmin(loss_differences)
        print(f"Token with max impact: index {max_diff_idx}, diff={loss_differences[max_diff_idx]:.6f}")
        print(f"Token with min impact: index {min_diff_idx}, diff={loss_differences[min_diff_idx]:.6f}")
            
        # ================================================================
        # 5) 绘制 token 网格热力图 (每个token一个格子，灰度表示差值)
        # ================================================================
        print(f"\n{'='*60}")
        print(f"=== PLOTTING: Token Grid Heatmap ===")
        print(f"{'='*60}")
        
        # 计算网格边长
        grid_size = int(np.ceil(np.sqrt(NUM_IMAGE_TOKENS)))
        print(f"Grid size: {grid_size}x{grid_size} (for {NUM_IMAGE_TOKENS} tokens)")
        
        # 将1D的loss差值转换为2D网格
        loss_diff_array = np.array(loss_differences)
        
        # 创建2D网格矩阵
        token_grid = np.zeros((grid_size, grid_size))
        
        # 将1D数组填充到2D网格中（按行优先）
        for idx, diff in enumerate(loss_differences):
            if idx < NUM_IMAGE_TOKENS:
                row = idx // grid_size
                col = idx % grid_size
                if row < grid_size and col < grid_size:
                    token_grid[row, col] = diff
        
        # 将原图转为 numpy 数组
        image_array = np.array(original_image)
        img_h, img_w = image_array.shape[:2]
        patch_h = img_h / grid_size
        patch_w = img_w / grid_size
        
        # 归一化 token_grid 到 [0, 1] 范围用于颜色映射
        if np.max(token_grid) != np.min(token_grid):
            token_grid_norm = (token_grid - np.min(token_grid)) / (np.max(token_grid) - np.min(token_grid))
        else:
            token_grid_norm = np.zeros_like(token_grid)
        
        # 使用 viridis colormap 将 token_grid 转为颜色
        viridis_cmap = plt.cm.viridis
        token_colors = viridis_cmap(token_grid_norm)[:, :, :3]  # 取 RGB，去掉 alpha
        
        # 图1: 纯原图
        plt.figure(figsize=(10, 8))
        plt.imshow(original_image)
        plt.title(f'Original Image\n({original_image.size[0]}x{original_image.size[1]})', fontsize=14)
        plt.xlabel('Width (pixels)', fontsize=12)
        plt.ylabel('Height (pixels)', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        
        original_image_path = f"{output_dir}/token_original.png"
        plt.savefig(original_image_path, dpi=150, bbox_inches='tight')
        print(f"Original image saved to: {original_image_path}")
        plt.close()
        
        # 图2: 使用viridis colormap (差值越大越亮)
        plt.figure(figsize=(10, 8))
        plt.imshow(token_grid, cmap='viridis', aspect='equal')
        plt.title(f'Loss Difference Grid (Viridis)\nGrid: {grid_size}x{grid_size}', fontsize=14)
        plt.xlabel('Column (Token Index % grid_size)', fontsize=12)
        plt.ylabel('Row (Token Index // grid_size)', fontsize=12)
        plt.colorbar(label='Loss Difference')
        plt.tight_layout()
        
        loss_grid_path = f"{output_dir}/token_loss_grid.png"
        plt.savefig(loss_grid_path, dpi=150, bbox_inches='tight')
        print(f"Loss grid plot saved to: {loss_grid_path}")
        plt.close()
        
        # 图3: 原图 + token 位置颜色叠加 (使用 viridis 颜色)
        overlay = np.zeros((img_h, img_w, 3))
        
        # 将每个 token 的颜色填充到对应的 patch 区域
        for row in range(grid_size):
            for col in range(grid_size):
                if row * grid_size + col < NUM_IMAGE_TOKENS:
                    y_start = int(row * patch_h)
                    y_end = int((row + 1) * patch_h)
                    x_start = int(col * patch_w)
                    x_end = int((col + 1) * patch_w)
                    overlay[y_start:y_end, x_start:x_end] = token_colors[row, col]
        
        # 将原图和叠加层混合 (alpha = 0.5)
        alpha = 0.5
        blended = (alpha * image_array / 255.0 + (1 - alpha) * overlay).astype(np.float32)
        blended = np.clip(blended, 0, 1)
        
        plt.figure(figsize=(10, 8))
        ax_blended = plt.gca()
        im = ax_blended.imshow(blended)
        ax_blended.set_title(f'Original Image + Token Colors\n(Viridis, alpha=0.5)', fontsize=14)
        ax_blended.set_xlabel('Image Width (pixels)', fontsize=12)
        ax_blended.set_ylabel('Image Height (pixels)', fontsize=12)
        plt.colorbar(im, label='Loss Difference')
        plt.tight_layout()
        
        blended_path = f"{output_dir}/token_blended.png"
        plt.savefig(blended_path, dpi=150, bbox_inches='tight')
        print(f"Blended image saved to: {blended_path}")
        plt.close()
        
        # 图4: TOP 50/100/200/300 tokens 渲染，其余位置用黑色
        # 按 loss 差值从大到小排序，获取索引
        loss_differences_array = np.array(loss_differences)
        sorted_indices = np.argsort(loss_differences_array)[::-1]  # 从大到小排序
        
        # 批量生成 TOP N 的可视化图
        for top_n in [25, 50, 100, 200, 300]:
            top_n_indices = sorted_indices[:top_n]  # TOP N
            
            # 创建黑色背景的叠加层
            overlay_black = np.zeros((img_h, img_w, 3))  # 黑色背景
            
            # 将TOP N tokens的位置用原图对应区域填充
            for token_idx in top_n_indices:
                if token_idx < NUM_IMAGE_TOKENS:
                    row = token_idx // grid_size
                    col = token_idx % grid_size
                    y_start = int(row * patch_h)
                    y_end = int((row + 1) * patch_h)
                    x_start = int(col * patch_w)
                    x_end = int((col + 1) * patch_w)
                    
                    # 从原图提取该区域的像素值（归一化到0-1）
                    overlay_black[y_start:y_end, x_start:x_end] = image_array[y_start:y_end, x_start:x_end] / 255.0
            
            plt.figure(figsize=(10, 8))
            ax_black = plt.gca()
            im_black = ax_black.imshow(overlay_black)
            ax_black.set_title(f'TOP {top_n} Tokens (from Original)\n(Black background)', fontsize=14)
            ax_black.set_xlabel('Image Width (pixels)', fontsize=12)
            ax_black.set_ylabel('Image Height (pixels)', fontsize=12)
            plt.tight_layout()
        
            top_black_path = f"{output_dir}/token_top{top_n}_black.png"
            plt.savefig(top_black_path, dpi=150, bbox_inches='tight')
            print(f"TOP {top_n} + Black saved to: {top_black_path}")
            plt.close()
        
        # 找到网格中差值最大的位置
        max_row, max_col = np.unravel_index(np.argmax(token_grid), token_grid.shape)
        max_token_idx = max_row * grid_size + max_col
        print(f"Max difference at: token_index={max_token_idx}, grid_pos=({max_row}, {max_col}), diff={token_grid[max_row, max_col]:.6f}")
        
        # 找到差值最小的位置
        min_row, min_col = np.unravel_index(np.argmin(token_grid), token_grid.shape)
        min_token_idx = min_row * grid_size + min_col
        print(f"Min difference at: token_index={min_token_idx}, grid_pos=({min_row}, {min_col}), diff={token_grid[min_row, min_col]:.6f}")
        
        print(f"\n{'='*60}")
        print(f"=== Finished processing sample {sample_image_index} ===")
        print(f"{'='*60}")
    

if __name__ == "__main__":
    train()
