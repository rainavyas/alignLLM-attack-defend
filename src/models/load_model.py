from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer

def load_model(core_args, device):
    if core_args.model_name == 'llama-2-7b-chat-hf':
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
        model, tokenizer = load_model_and_tokenizer(model_path, 
                       low_cpu_mem_usage=True, 
                       use_cache=False,
                       device=device)
        return model, tokenizer
    else:
        raise NotImplementedError