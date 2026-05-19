import os
import re
import abc
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from sven.model import load_model
from sven.constant import PROMPTS
from sven.utils import try_parse, set_seed_

from cosec.CustomizedGeneration import CodeLlamaModelLM, Qwen2ModelLM
from sven.constant import MODEL_DIRS


def truncate_after(completion, trunc_str):
    return completion[:completion.find(trunc_str) + len(trunc_str)]


def truncate_before(completion, trunc_str):
    return completion[:completion.find(trunc_str)].rstrip()


def truncate_after_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str) + len(trunc_str)]


def truncate_before_last(completion, trunc_str):
    return completion[:completion.rfind(trunc_str)]


class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.load_model()
    
    def update_args(self, args):
        self.args = args
    
    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()
    
    @abc.abstractclassmethod
    def sample(self, file_context, func_context, info):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def preprocess(self, file_context, func_context, info):
        raise NotImplementedError()

    def postprocess(self, completion, info):
        if info['language'] == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                if '\n    #' in completion:
                    completion = truncate_before_last(completion, '\n    #')
        elif info['language'] in ['c', 'cpp']:
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n}'
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'go':
            if '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'js':
            if '\n});' in completion: # for app function definitions
                completion = truncate_after(completion, '\n});')
            elif re.search(r'\n}(?!;)', completion) is not None: # normal function end
                match = re.search(r'\n}(?!;)', completion)
                completion = completion[:match.end()]
            elif '\n//' in completion:
                completion = truncate_before_last(completion, '\n//').rstrip()
            elif '\n/*' in completion:
                completion = truncate_before_last(completion, '\n/*').rstrip()
            elif '\n    //' in completion:
                completion = truncate_before_last(completion, '\n    //').rstrip() + '\n}'
            elif '\n    /*' in completion:
                completion = truncate_before_last(completion, '\n    /*').rstrip() + '\n}'
            else:
                completion = completion
        elif info['language'] == 'jsx':
            # only for cwe-200 0-jsx
            if '\n' in completion:
                completion = truncate_before(completion, '\n')
        elif info['language'] == 'rb':
            if '\n    end' in completion:
                completion = truncate_after(completion, '\n    end') + '\nend'
            elif '\nend' in completion:
                completion = truncate_after(completion, '\nend')
            elif '    #' in completion:
                completion = truncate_before_last(completion, '    #').rstrip('\n') + '\nend'
                if '\nend' not in completion: completion += '\nend'
            else:
                completion = completion
        elif info['language'] == 'java':
            if '\n    }' in completion:
                completion = truncate_after(completion, '\n    }') + '\n}'
            elif '\n}' in completion:
                completion = truncate_after(completion, '\n}')
            elif ';\n' in completion:
                completion = truncate_after_last(completion, ';\n') + '\n    }' + '\n}'
            elif '    //' in completion:
                completion = truncate_before_last(completion, '    //').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            elif '    /*' in completion:
                completion = truncate_before_last(completion, '    /*').rstrip('\n') + '\n}'
                if '\n}' not in completion: completion += '\n}'
            else:
                completion = completion
        else:
            raise NotImplementedError('Postprocessing for {language} is not implemented yet'.format(language=info['language']))

        if 'postprocess' in info:
            scope = {'completion': completion}
            exec(info['postprocess'], scope)
            completion = scope['completion']

        return completion


class LMEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context
    
    def sample(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.input_device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed_(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs
    

class PrefixEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.model_dir, False, self.args)
        self.model.eval()

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context

    def sample(self, file_context, func_context, info):
        control = 0
        return self.sample_prefix(file_context, func_context, control, info)
    
    def sample_prefix(self, file_context, func_context, control, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.input_device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed_(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                control_id=control,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)

        return output_srcs, non_parsed_srcs
    

class DeepGuardEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        from deepguard.inference import load_model
        self.tokenizer, self.model, self.input_device = load_model('deepguard', self.args.model_dir, False, self.args)
        self.model.eval()

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context

    def sample(self, file_context, func_context, info):
        return self.sample_guard(file_context, func_context, info)

    def sample_guard(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.input_device)
        attention_mask = inputs['attention_mask'].to(self.input_device) if 'attention_mask' in inputs else None
        input_ids_len = input_ids.size(1)

        output_srcs, non_parsed_srcs = [], []
        
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed_(self.args.seed + i)
            with torch.no_grad():
                gen_output = self.model.generate_with_security(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    num_return_sequences=self.args.num_samples_per_gen,
                    temperature=self.args.temp,
                    max_new_tokens=self.args.max_gen_len,
                    top_p=self.args.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs
    

class CoSecEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        if self.args.model_name in MODEL_DIRS:
            self.args.model_name = MODEL_DIRS[self.args.model_name]
        if self.args.base_model in MODEL_DIRS:
            self.args.base_model = MODEL_DIRS[self.args.base_model]
            print(self.args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model, local_files_only=True)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.bos_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if 'deepseek' in str(self.args.model_name).lower() or 'seed' in str(self.args.model_name).lower():
            self.model = CodeLlamaModelLM.from_pretrained(self.args.model_name, device_map='auto', local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(self.args.base_model, device_map='auto', local_files_only=True)
            base_model.resize_token_embeddings(len(tokenizer))
            self.sec_model = PeftModel.from_pretrained(base_model, self.args.sec_model)
        elif 'qwen' in str(self.args.model_name).lower():
            self.model = Qwen2ModelLM.from_pretrained(self.args.model_name, device_map='auto', local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(self.args.base_model, device_map='auto', local_files_only=True)
            base_model.resize_token_embeddings(len(tokenizer))
            self.sec_model = PeftModel.from_pretrained(base_model, self.args.sec_model)
        else:
            raise NotImplementedError()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, local_files_only=True)
        self.model.eval()
        self.sec_model.eval()
        self.input_device = self.model.device

    def preprocess(self, file_context, func_context, info):
        return file_context + func_context

    def sample(self, file_context, func_context, info):
        return self.sample_cosec(file_context, func_context, info)
    
    def sample_cosec(self, file_context, func_context, info):
        prompt = self.preprocess(file_context, func_context, info)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.input_device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed_(self.args.seed + i)
            kwargs = {
                'expert': True,
                'expert_lm': self.sec_model,
                'model_kwargs_expert': {},
                'threshold': self.args.threshold,
            }
            gen_output = self.model.generate_with_experts(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                expert_min_prob=0.0,
                expert_temperature=self.args.exp_temp,
                expert_top_p=0.95,
                **kwargs
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs


class TextPromptEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def sample(self, file_context, func_context, info):
        control = 0
        return self.sample_text(file_context, func_context, control, info)

    def sample_text(self, file_context, func_context, control, info):
        if info['language'] == 'py':
            input_src = file_context + '# ' + PROMPTS[control] + func_context
        elif info['language'] == 'c':
            input_src = file_context + '// ' + PROMPTS[control] + func_context
        else:
            raise NotImplementedError()
        prompt = input_src
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.input_device)
        input_ids_len = input_ids.size(1)
        output_srcs, non_parsed_srcs = [], []
        for i in range(self.args.num_samples // self.args.num_samples_per_gen):
            set_seed_(self.args.seed + i)
            gen_output = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=self.args.num_samples_per_gen,
                temperature=self.args.temp,
                max_new_tokens=self.args.max_gen_len,
                top_p=self.args.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            tokens = gen_output[:, input_ids_len:, ...]
            completions = self.tokenizer.batch_decode(tokens)
            for completion in completions:
                if self.tokenizer.eos_token in completion:
                    completion = completion[:completion.find(self.tokenizer.eos_token)]
                completion = self.postprocess(completion, info)
                output_src = file_context + func_context + completion
                output_src = output_src.rstrip() + '\n'
                if info['language'] != 'go' and try_parse(output_src, info) != 0:
                    non_parsed_srcs.append(output_src)
                else:
                    output_srcs.append(output_src)
        return output_srcs, non_parsed_srcs
