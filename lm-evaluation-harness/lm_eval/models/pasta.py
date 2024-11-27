import copy
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from lm_eval.models.huggingface import HFLM

from pastalib.pasta import PASTA
from pastalib.utils.CustomLlama3Tokenizer import CustomLlama3Tokenizer

eval_logger = utils.eval_logger

@register_model("hf-pasta")
class PastaLM(HFLM):
    PASTA_HEAD_CONFIGS = {
        "tinyllama": {k: None for k in range(22)},
        "llama-2": {k: None for k in range(32)},
        "llama-3": {k: None for k in range(32)},
    }

    PASTA_HEAD_CONFIGS = {
        "tinyllama": {k: None for k in range(22)},
        "llama-2": {k: None for k in range(32)},
        "llama-3": {0: [15], 1: [8], 2: [15], 3: [0], 4: [16], 5: [4], 6: [6], 7: [25], 8: [11], 9: [3], 10: [13], 11: [13], 12: [15], 13: [18], 14: [22], 15: [30], 16: [1], 17: [29], 18: [20], 19: [9], 20: [14], 21: [26], 22: [8], 23: [22], 24: [27], 25: [5],26: [30],27: [6], 28: [0], 29: [16], 30:[2], 31:[21]},
    }
    
    @staticmethod
    def get_pasta_head_config(model_name):
        for k, v in PastaLM.PASTA_HEAD_CONFIGS.items():
            if k in model_name.lower():
                return v
        return None
    
    def __init__(self, *args, alpha=None, **kwargs):
        super().__init__(*args, **kwargs)

        if 'llama-3' in self.pretrained.lower():
            self.tokenizer = CustomLlama3Tokenizer(self.tokenizer)
            warnings.warn("Currently using hot-patched llama-3 tokenizer. Update to officiel tokenizer when offset_mapping bug is fixed.")
        
        assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
        assert alpha is not None
        self.alpha = alpha
        
        assert isinstance(self.pretrained, str)
        self.head_config = self.get_pasta_head_config(self.pretrained)
        if self.head_config is None:
            raise ValueError(f"Pasta head config not found for {self.pretrained}. Populate PASTA_HEAD_CONFIGS with retrieval heads for this model before running HayStack with Pasta intervention.")
            
        # Initialize the PASTA steerer
        self.pasta = PASTA(
            model=self.model,
            tokenizer=self.tokenizer,
            head_config=self.head_config, 
            alpha=self.alpha,
            scale_position="exclude",
        )
    
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None, return_offsets_mapping=False,
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        # encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        tokenized_outputs = self.tokenizer._encode_plus(string, return_offsets_mapping=return_offsets_mapping,
                                                        **special_tokens_kwargs)
        encoding = tokenized_outputs['input_ids']

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        if return_offsets_mapping:
            return encoding, tokenized_outputs['offset_mapping']
        else:
            return encoding
    
    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc, offset_mapping = self.tok_encode(context + continuation, add_special_tokens=False, 
                                                    return_offsets_mapping=True) # this would get the full offset_mappings
        context_enc = self.tok_encode(context, add_special_tokens=False, return_offsets_mapping=False)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc, offset_mapping
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # extract raw user context before wrap_chat_template contaminates
        user_contexts = [req.args[0] for req in requests]

        if self.use_chat_template:
            print(f"First element before prompt formatting...\n{requests[0].args}")
            requests = self.wrap_chat_template(requests)
            print(f"First element after prompt formatting...\n{requests[0].args}")

        assert len(requests) == len(user_contexts)
        new_reqs = []
        for req, user_context in zip(requests, user_contexts):
            context, continuation = req.args
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
                offset_mapping = None
            else:
                context_enc, continuation_enc, offset_mapping = self._encode_pair(context, continuation)
            new_req = (((context, continuation, user_context), offset_mapping, context_enc, continuation_enc))
            new_reqs.append(new_req)

        return self._loglikelihood_tokens(new_reqs)
    
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str, str], List[Tuple[int, int]], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req: Tuple[Tuple[str, str, str], List[Tuple[int, int]], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[-2] + req[-1]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str, str], List[Tuple[int, int]], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []
            contexts = []
            
            full_prompts = []
            user_contexts = []
            offset_mappings = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for (context, continuation, user_context), offset_mapping, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                else:
                    raise NotImplementedError(f"Only decoder-only transformer is supported.")

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )
                
                full_prompt = context + continuation
                full_prompts.append(full_prompt)
                
                user_contexts.append(user_context)

                offset_mappings.append(offset_mapping)
                
                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
                contexts.append(context)

            # create encoder attn mask and batched conts, if seq2seq
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            else:
                raise NotImplementedError(f"Only decoder-only transformer is supported.")
                
            outputs = self._model_call(batched_inps, full_prompts, user_contexts, offset_mappings)

            multi_logits = F.log_softmax(outputs, dim=-1)  # [batch, padding_length (inp or cont), vocab]

            for (request_str, _, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # trim user context to revert to original behavior for caching
                request_str = request_str[:-1]
                assert len(request_str) == 2
                
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)
    
    def _model_call(self, inps, full_prompts, emphasized_texts, offset_mappings, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        model_input = {'input_ids': inps}
        
        # PASTA registers the pre_forward_hook to edit attention
        with self.pasta.apply_steering(
            model=self.model, 
            strings=full_prompts, 
            substrings=emphasized_texts, 
            model_input=model_input,
            offsets_mapping=offset_mappings,
        ) as steered_model: 
            with torch.no_grad():
                return steered_model(**model_input).logits
            
    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=return_offsets_mapping,
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        if return_offsets_mapping:
            return encoding["input_ids"], encoding["attention_mask"], encoding["offset_mapping"]
        else:
            return encoding["input_ids"], encoding["attention_mask"]
            
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:

        # extract raw user context before wrap_chat_template contaminates
        user_contexts = [req.args[0] if isinstance(req.args[0], str) else req.args[0][0] for req in requests]
        
        if self.use_chat_template:
            print(f"First element before prompt formatting...\n{requests[0].args}")
            requests = self.wrap_chat_template(requests)
            print(f"First element after prompt formatting...\n{requests[0].args}")
        
        for req, user_context in zip(requests, user_contexts):
            req.args = (*req.args, user_context)

        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs, user_contexts = zip(*chunk)
            
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks, offset_mappings = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
                return_offsets_mapping=True,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                offset_mappings=offset_mappings,
                full_prompts=contexts,
                emphasized_texts=user_contexts,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res
    
    def _model_generate(self, context, max_length, stop, offset_mappings, 
                        full_prompts, emphasized_texts, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        model_input = {'input_ids': context}
        
        with self.pasta.apply_steering(
            model=self.model, 
            strings=full_prompts,
            substrings=emphasized_texts, 
            model_input=model_input,
            offsets_mapping=offset_mappings,
        ) as steered_model: 
            # print(inputs['input_ids'].shape)
            outputs = steered_model.generate(
                **model_input,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
        
        '''
        outputs = self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
        '''
        
        return outputs
