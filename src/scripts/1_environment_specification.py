import os, sys

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}/")
import json
import fire
import argparse
from utils.description_evolve import DescEvolve
import random
from utils.data_utils import parse_list_by_n
# random.seed(42)


def parse():
    parser = argparse.ArgumentParser()

    # Fill or Path
    parser.add_argument("--api_keys_file", type=str, default="key.txt")
    # 使用 Azure 的封装接口（企业版 GPT 服务） 或 直接访问 OpenAI 的模型（如 GPT-3.5、GPT-4、GPT-4o）
    parser.add_argument(
        "--api_type", type=str, choices=["azure", "openai", "mix"], default="azure"
    )
    parser.add_argument("--prompt_file", type=str, default="prompt/desc_evol")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--context_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--example_num", type=int, default=1)

    # Methodology Config 最大迭代修正次数
    parser.add_argument("--max_correction", type=int, default=3)

    # GPT options
    parser.add_argument("--model", type=str, default="gpt-4-0125-preview")
    parser.add_argument(
        "--stop_tokens", type=str, default="\n\n", help="Split stop tokens by ||"
    )
    parser.add_argument("--n_process", type=int)

    # debug options 挺反常规。。 不写 -v是True
    parser.add_argument("-v", "--verbose", action="store_false")

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split("||")
    args.prompt_file = f"{ROOT_DIR}/{args.prompt_file}"
    args.output_path = f"{ROOT_DIR}/{args.output_path}"
    args.data_path = f"{ROOT_DIR}/{args.data_path}"
    args.context_path = f"{ROOT_DIR}/{args.context_path}"
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


# Multi-process (Optional)
def main(args):
    data = json.load(open(args.data_path))
    all_contexts = json.load(open(args.context_path))
    context_list_list = parse_list_by_n(lst=all_contexts, n=args.n_process)
    result = []
    for context_list in context_list_list:
        desc_evolver = DescEvolve(args)
        # 为每个context_list采集example
        examples_list = [random.sample(data, k=args.example_num) for _ in context_list]
        examples_list = [
            [_["description"] for _ in examples] for examples in examples_list
        ]
        # 构造prompt并调用GPT
        # 输出为len(context_list)个{'description': description, 'time': time.time() - st, 'token': total_token}组成的列表  description就是GPT输出
        generated_items = desc_evolver.close_loop_evol_multiprocess(
            [context["instruction"] for context in context_list], examples_list
        )
        for generated_item, context in zip(generated_items, context_list):
            generated_item.update({"evol_from": context})
        # description就是GPT输出
        generated_items = [_ for _ in generated_items if _["description"] is not None]
        result += generated_items
    json.dump(obj=result, fp=open(args.output_path, "w", encoding="utf"), indent=4)


if __name__ == "__main__":
    # fire.Fire(main)
    args = parse()
    main(args)
