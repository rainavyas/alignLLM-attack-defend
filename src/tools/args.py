import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='llama-2-7b-chat-hf', help='aligned LLM')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='advbench', help='dataset for exps')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()


def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--attack_method', type=str, default='gcg', choices=['gcg'], help='Adversarial attack approach')
    commandLineParser.add_argument('--topk', type=int, default=256, help='topk candidates for gcg')
    commandLineParser.add_argument('--batch_size', type=int, default=512, help='batchsize for gcg alg')
    commandLineParser.add_argument('--adv_init_string', type=str, default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", help='initialisation string for gcg')
    commandLineParser.add_argument('--steps', type=int, default=500, help='iter steps for gcg alg')
    commandLineParser.add_argument('--start', type=int, default=0, help='data samples start')
    commandLineParser.add_argument('--end', type=int, default=10000, help='data samples end')
    return commandLineParser.parse_known_args()