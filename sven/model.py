import os
import torch
from typing import Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoConfig, logging, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from sven.hf import GPT2CustomConfig
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache


class CodeLlamaPrefix(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_key_value_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):
                    param = torch.nn.Parameter(
                        torch.zeros(
                            config.num_key_value_heads,
                            config.n_prefix_token,
                            head_dim
                        ), requires_grad=True
                    )
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = []
        for layer in range(self.config.num_hidden_layers):
            key_list, val_list = [], []
            for cid in control_ids:
                idx = cid * self.config.num_hidden_layers * 2 + layer * 2
                key_list.append(self.dropout(self.prefix_params[idx]))
                val_list.append(self.dropout(self.prefix_params[idx + 1]))
            past.append((torch.stack(key_list), torch.stack(val_list)))
        return tuple(past)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            # Calculate the actual past length excluding prefix tokens
            past_len = past_key_values[0][0].shape[-2]
            prefix_len = self.config.n_prefix_token
            
            # Only truncate if we've processed more than just the prefix
            if past_len > prefix_len:
                actual_processed = past_len - prefix_len
                if actual_processed < input_ids.shape[1]:
                    input_ids = input_ids[:, actual_processed:]
                else:
                    # Keep only the last token
                    input_ids = input_ids[:, -1:]
                    if input_ids.shape[1] == 0:
                        input_ids = input_ids[:, -1].unsqueeze(-1)
            # else: First step after prefix init, keep all input_ids
        else:
            control_ids = [kwargs.get('control_id', 0)] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
        
        return {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": None,
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                cache_position,
            )


class SeedCoderPrefix(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):
                    param = torch.nn.Parameter(
                        torch.zeros(
                            config.num_key_value_heads,
                            config.n_prefix_token,
                            head_dim
                        ), requires_grad=True
                    )
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = []
        for layer in range(self.config.num_hidden_layers):
            key_list, val_list = [], []
            for cid in control_ids:
                idx = cid * self.config.num_hidden_layers * 2 + layer * 2
                key_list.append(self.dropout(self.prefix_params[idx]))
                val_list.append(self.dropout(self.prefix_params[idx + 1]))
            past.append((torch.stack(key_list), torch.stack(val_list)))
        return tuple(past)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            # Calculate the actual past length excluding prefix tokens
            past_len = past_key_values[0][0].shape[-2]
            prefix_len = self.config.n_prefix_token
            
            # Only truncate if we've processed more than just the prefix
            if past_len > prefix_len:
                actual_processed = past_len - prefix_len
                if actual_processed < input_ids.shape[1]:
                    input_ids = input_ids[:, actual_processed:]
                else:
                    # Keep only the last token
                    input_ids = input_ids[:, -1:]
                    if input_ids.shape[1] == 0:
                        input_ids = input_ids[:, -1].unsqueeze(-1)
            # else: First step after prefix init, keep all input_ids
        else:
            control_ids = [kwargs.get('control_id', 0)] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)
        
        return {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": None,
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                cache_position,
            )


class Qwen2Prefix(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):
                    param = torch.nn.Parameter(
                        torch.zeros(
                            config.num_key_value_heads,
                            config.n_prefix_token,
                            head_dim
                        ), requires_grad=True
                    )
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

        self._prefix_cache = {}

    def get_past_from_prefix(self, control_ids):
        past = []
        for layer in range(self.config.num_hidden_layers):
            key_list, val_list = [], []
            for cid in control_ids:
                idx = cid * self.config.num_hidden_layers * 2 + layer * 2
                key_list.append(self.dropout(self.prefix_params[idx]))
                val_list.append(self.dropout(self.prefix_params[idx + 1]))
            past.append((torch.stack(key_list), torch.stack(val_list)))
        return past

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, 
                                    inputs_embeds=None, cache_position=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        # Handle cache compatibility
        if past_key_values is not None:
            # Support for Cache objects (new transformers)
            if isinstance(past_key_values, Cache):
                # Check if cache is empty
                if past_key_values.get_seq_length() == 0:
                    # Initialize with prefix
                    control_ids = [kwargs.get('control_id', 0)] * batch_size
                    past_key_values = self.get_past_from_prefix(control_ids)
                    
                    # Create attention mask that accounts for prefix tokens
                    if attention_mask is None:
                        attention_mask = torch.ones((batch_size, seq_length + self.config.n_prefix_token), 
                                                  dtype=torch.long, device=input_ids.device)
                    else:
                        # Prepend 1s for prefix tokens
                        prefix_mask = torch.ones((batch_size, self.config.n_prefix_token), 
                                               dtype=attention_mask.dtype, device=attention_mask.device)
                        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                else:
                    past_length = past_key_values.get_seq_length()
                    
                    # Adjust input_ids to only include new tokens
                    if past_length >= self.config.n_prefix_token:
                        actual_past_length = past_length - self.config.n_prefix_token
                        if actual_past_length < seq_length:
                            input_ids = input_ids[:, actual_past_length:]
                        else:
                            input_ids = input_ids[:, -1:]
                            if input_ids.shape[1] == 0:
                                input_ids = input_ids[:, -1].unsqueeze(-1)
                    
                    # Attention mask should cover all tokens including past
                    if attention_mask is not None:
                        # Ensure attention mask covers the full sequence length
                        full_seq_length = past_length + input_ids.shape[1]
                        if attention_mask.shape[1] < full_seq_length:
                            # Extend attention mask
                            extension = torch.ones((batch_size, full_seq_length - attention_mask.shape[1]), 
                                                 dtype=attention_mask.dtype, device=attention_mask.device)
                            attention_mask = torch.cat([attention_mask, extension], dim=1)
            else:
                # Legacy list/tuple format
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    past_length = past_key_values[0][0].shape[2]
                    
                    # If this is the first call with prefix, adjust attention mask
                    if past_length == self.config.n_prefix_token and seq_length > 0:
                        if attention_mask is None:
                            attention_mask = torch.ones((batch_size, seq_length + self.config.n_prefix_token), 
                                                      dtype=torch.long, device=input_ids.device)
                        else:
                            # Prepend 1s for prefix tokens
                            prefix_mask = torch.ones((batch_size, self.config.n_prefix_token), 
                                                   dtype=attention_mask.dtype, device=attention_mask.device)
                            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                    else:
                        # Adjust input_ids for subsequent steps
                        if past_length >= self.config.n_prefix_token:
                            actual_past_length = past_length - self.config.n_prefix_token
                            if actual_past_length < seq_length:
                                input_ids = input_ids[:, actual_past_length:]
                            else:
                                input_ids = input_ids[:, -1:]
                                if input_ids.shape[1] == 0:
                                    input_ids = input_ids[:, -1].unsqueeze(-1)
                        
                        # Ensure attention mask covers full sequence
                        if attention_mask is not None:
                            full_seq_length = past_length + input_ids.shape[1]
                            if attention_mask.shape[1] < full_seq_length:
                                extension = torch.ones((batch_size, full_seq_length - attention_mask.shape[1]), 
                                                     dtype=attention_mask.dtype, device=attention_mask.device)
                                attention_mask = torch.cat([attention_mask, extension], dim=1)
                else:
                    # First generation step - initialize prefix
                    control_ids = [kwargs.get('control_id', 0)] * batch_size
                    past_key_values = self.get_past_from_prefix(control_ids)
                    
                    # Create attention mask with prefix
                    if attention_mask is None:
                        attention_mask = torch.ones((batch_size, seq_length + self.config.n_prefix_token), 
                                                  dtype=torch.long, device=input_ids.device)
                    else:
                        prefix_mask = torch.ones((batch_size, self.config.n_prefix_token), 
                                               dtype=attention_mask.dtype, device=attention_mask.device)
                        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            # Initialize with prefix
            control_ids = [kwargs.get('control_id', 0)] * batch_size
            past_key_values = self.get_past_from_prefix(control_ids)
            
            # Create attention mask with prefix
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length + self.config.n_prefix_token), 
                                          dtype=torch.long, device=input_ids.device)
            else:
                prefix_mask = torch.ones((batch_size, self.config.n_prefix_token), 
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Note: Qwen2 doesn't use cache_position in the parent's forward method
        return {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )


# Keep the rest of the functions unchanged
def model_from_pretrained(lm_path, model_type, config):
    kwargs = dict()

    # loading model from local cache
    if lm_path.startswith('deepseek-ai/deepseek-coder-'):
        if model_type == 'lm':
            model_class = LlamaForCausalLM
        elif model_type == 'prefix':
            model_class = CodeLlamaPrefix
        else:
            assert False
    elif lm_path.startswith('codellama/CodeLlama-'):
        if model_type == 'lm':
            model_class = LlamaForCausalLM 
        elif model_type == 'prefix':
            model_class = CodeLlamaPrefix
        else:
            assert False
    elif lm_path.startswith('ByteDance-Seed/Seed-'):
        if model_type == 'lm':
            model_class = LlamaForCausalLM 
        elif model_type == 'prefix':
            model_class = SeedCoderPrefix
        else:
            assert False
    elif lm_path.startswith('Qwen/Qwen2.5-Coder-'):
        if model_type == 'lm':
            model_class = Qwen2ForCausalLM
        elif model_type == 'prefix':
            model_class = Qwen2Prefix
        else:
            assert False
    else:
        # Local path or unrecognised HF repo — let AutoModelForCausalLM
        # read config.json to pick the right class.
        assert model_type == 'lm', f"prefix tuning not supported for local path: {lm_path}"
        model_class = AutoModelForCausalLM
    if config is None:
        model = model_class.from_pretrained(lm_path, local_files_only=True, **kwargs)
    else:
        model = model_class.from_pretrained(lm_path, local_files_only=True, **kwargs, config=config)

    return model

def config_from_pretrained(lm_path, path):
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    else:
        return AutoConfig.from_pretrained(path, local_files_only=True)

def save_model(model, path, args):
    if type(model) in (CodeLlamaPrefix, Qwen2Prefix, SeedCoderPrefix):
        assert args.pretrain_dir.startswith('deepseek-ai/deepseek-coder-') or args.pretrain_dir.startswith('Qwen/Qwen2.5-Coder-') \
             or args.pretrain_dir.startswith('codellama/CodeLlama-') or args.pretrain_dir.startswith('ByteDance-Seed/Seed-')
        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)

def load_model(model_type, path, is_training, args):
    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type == 'lm':
        config = config_from_pretrained(path, path)
        model = model_from_pretrained(path, model_type, config)
    elif model_type == 'prefix':
        if is_training:
            lm_path = path
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = args.n_prefix_token
            lm_config.prefix_dropout = args.dropout
            lm_config.n_control = 2
            model = model_from_pretrained(lm_path, model_type, lm_config)
        else:
            lm_path_file = os.path.join(path, 'lm.txt')
            assert os.path.exists(lm_path_file)
            with open(lm_path_file) as f:
                lm_path = f.read()
            prefix_config = config_from_pretrained(lm_path, path)
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = prefix_config.n_prefix_token
            lm_config.prefix_dropout = prefix_config.prefix_dropout
            lm_config.n_control = prefix_config.n_control
            model = model_from_pretrained(lm_path, model_type, lm_config)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            model.prefix_params.load_state_dict(torch.load(prefix_file))
    else:
        assert False

    model.resize_token_embeddings(len(tokenizer))
    input_device = parallelize_model(model, args)
    return tokenizer, model, input_device

def parallelize_model(model, args):
    if args.n_gpu > 1:
        model.parallelize()
        if 'codegen' in str(args.model_dir).lower():
            input_device = model.transformer.first_device
        elif 'qwen2.5' in str(args.model_dir).lower():
            input_device = model.first_device
    else:
        model.to(args.device)
        input_device = args.device
    return input_device