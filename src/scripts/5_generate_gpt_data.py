import os, sys

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}/")
import json
import fire
from utils.pddlgym_utils import gen_traj_batch
from utils.data_utils import traj2gpt_wonl, traj2gpt_wonl_open_loop


def main(data_path, output_path, solvable_path=None, close_loop=True):
    """
    将 PDDL domain + problem 文件求解成执行轨迹，
    并进一步转换为 GPT 可训练的数据格式。

    参数说明：
    - data_path: 输入 JSON 文件路径（通常是上一步 4_generate_problems 的输出）
    - output_path: GPT 格式数据输出路径
    - solvable_path: 可选，保存“可被规划求解”的 domain/problem
    - close_loop: 是否采用“闭环”数据格式（包含自然语言接口）
    """
    # 使用的规划算法（Fast Downward 的经典启发式）
    algo_list = ["seq-opt-lmcut"]
    data = json.load(open(data_path))
    solvable_envs = []  # 保存可求解环境
    gpt_list = []  # 保存可求解环境
    for item in data:
        domain, problems = item["domain"], item["problems"]
        # 使用规划器（如 Fast Downward）生成执行轨迹
        # predicate_map 是一个 PDDL 名称到自然语言模板的映射表
        traj_data = gen_traj_batch(
            domain, problems, algo_list, predicate_map=item["nl_interface"]
        )
        # 筛选出可以运行的一组轨迹 第一组19条 有5条可以运行
        if len(traj_data) > 0 and solvable_path is not None:  # 可规划求解的 (solvable)
            solvable_envs.append(item)
        if close_loop:  # 使用自然语言接口（NL interface）将每步动作转换为自然语言描述
            for idx, traj in enumerate(traj_data):
                # 构成gpt的对话结构 一句observe 一句action
                gpt_data = traj2gpt_wonl(item, traj, idx)
                gpt_list.append(gpt_data)
        else:
            for idx, traj in enumerate(traj_data):
                gpt_data = traj2gpt_wonl_open_loop(item, traj, idx)
                gpt_list.append(gpt_data)

    json.dump(obj=gpt_list, fp=open(output_path, "w", encoding="utf"), indent=4)
    if solvable_path is not None:
        json.dump(
            obj=solvable_envs, fp=open(solvable_path, "w", encoding="utf"), indent=4
        )


if __name__ == "__main__":
    fire.Fire(main)
