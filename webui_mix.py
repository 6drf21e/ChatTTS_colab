import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import re
import time

import pandas
import numpy as np
from tqdm import tqdm
import random
import gradio as gr
import json
from utils import batch_split, normalize_zh
from tts_model import load_chat_tts_model, clear_cuda_cache, generate_audio_for_seed
from config import DEFAULT_BATCH_SIZE, DEFAULT_SPEED, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_ORAL, \
    DEFAULT_LAUGH, DEFAULT_BK, DEFAULT_SEG_LENGTH

parser = argparse.ArgumentParser(description="Gradio ChatTTS MIX")
parser.add_argument("--source", type=str, default="huggingface", help="Model source: 'huggingface' or 'local'.")
parser.add_argument("--local_path", type=str, help="Path to local model if source is 'local'.")
parser.add_argument("--share", default=False, action="store_true", help="Share the server publicly.")

args = parser.parse_args()

# 存放音频种子文件的目录
SAVED_DIR = "saved_seeds"

# mkdir
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

# 文件路径
SAVED_SEEDS_FILE = os.path.join(SAVED_DIR, "saved_seeds.json")

# 选中的种子index
SELECTED_SEED_INDEX = -1

# 初始化JSON文件
if not os.path.exists(SAVED_SEEDS_FILE):
    with open(SAVED_SEEDS_FILE, "w") as f:
        f.write("[]")

chat = load_chat_tts_model(source=args.source, local_path=args.local_path)
# chat = None
# chat = load_chat_tts_model(source="local", local_path=r"models")

# 抽卡的最大数量
max_audio_components = 10


# print("loading ChatTTS model...")
# chat = ChatTTS.Chat()
# chat.load_models(source="local", local_path="models")
# torch.cuda.empty_cache()


# 加载
def load_seeds():
    with open(SAVED_SEEDS_FILE, "r") as f:
        global saved_seeds

        seeds = json.load(f)

        # 兼容旧的 JSON 格式，添加 path 字段
        for seed in seeds:
            if 'path' not in seed:
                seed['path'] = None

        saved_seeds = seeds
    return saved_seeds


def display_seeds():
    seeds = load_seeds()
    # 转换为 List[List] 的形式
    return [[i, s['seed'], s['name'], s['path']] for i, s in enumerate(seeds)]


saved_seeds = load_seeds()
num_seeds_default = 2


def save_seeds():
    global saved_seeds
    with open(SAVED_SEEDS_FILE, "w") as f:
        json.dump(saved_seeds, f)
    saved_seeds = load_seeds()


# 添加 seed
def add_seed(seed, name, audio_path, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            return False
    saved_seeds.append({
        'seed': seed,
        'name': name,
        'path': audio_path
    })
    if save:
        save_seeds()


# 修改 seed
def modify_seed(seed, name, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            s['name'] = name
            if save:
                save_seeds()
            return True
    return False


def delete_seed(seed, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            saved_seeds.remove(s)
            if save:
                save_seeds()
            return True
    return False


def generate_seeds(num_seeds, texts, tq):
    """
    生成随机音频种子并保存
    :param num_seeds:
    :param texts:
    :param tq:
    :return:
    """
    seeds = []
    sample_rate = 24000
    # 按行分割文本 并正则化数字和标点字符
    texts = [normalize_zh(_) for _ in texts.split('\n') if _.strip()]
    print(texts)
    if not tq:
        tq = tqdm
    for _ in tq(range(num_seeds), desc=f"随机音色生成中..."):
        seed = np.random.randint(0, 9999)

        filename = generate_audio_for_seed(chat, seed, texts, 1, 5, "[oral_2][laugh_0][break_4]", None, 0.3, 0.7, 20)
        seeds.append((filename, seed))
        clear_cuda_cache()

    return seeds


# 保存选定的音频种子
def do_save_seed(seed, audio_path):
    print(f"Saving seed {seed} to {audio_path}")
    seed = seed.replace('保存种子 ', '').strip()
    if not seed:
        return
    add_seed(int(seed), seed, audio_path)
    gr.Info(f"Seed {seed} has been saved.")


def do_save_seeds(seeds):
    assert isinstance(seeds, pandas.DataFrame)

    seeds = seeds.drop(columns=['Index'])

    # 将 DataFrame 转换为字典列表格式，并将键转换为小写
    result = [{k.lower(): v for k, v in row.items()} for row in seeds.to_dict(orient='records')]
    print(result)
    if result:
        global saved_seeds
        saved_seeds = result
        save_seeds()
        gr.Info(f"Seeds have been saved.")
    return result


def do_delete_seed(val):
    # 从 val 匹配 [(\d+)] 获取index
    index = re.search(r'\[(\d+)\]', val)
    global saved_seeds
    if index:
        index = int(index.group(1))
        seed = saved_seeds[index]['seed']
        delete_seed(seed)
        gr.Info(f"Seed {seed} has been deleted.")
    return display_seeds()


# 定义播放音频的函数
def do_play_seed(val):
    # 从 val 匹配 [(\d+)] 获取index
    index = re.search(r'\[(\d+)\]', val)
    if index:
        index = int(index.group(1))
        seed = saved_seeds[index]['seed']
        audio_path = saved_seeds[index]['path']
        if audio_path:
            return gr.update(visible=True, value=audio_path)
    return gr.update(visible=False, value=None)


def seed_change_btn():
    global SELECTED_SEED_INDEX
    if SELECTED_SEED_INDEX == -1:
        return ['删除', '试听']
    return [f'删除 idx=[{SELECTED_SEED_INDEX[0]}]', f'试听 idx=[{SELECTED_SEED_INDEX[0]}]']


def audio_interface(num_seeds, texts, progress=gr.Progress()):
    """
    生成音频
    :param num_seeds:
    :param texts:
    :param progress:
    :return:
    """
    seeds = generate_seeds(num_seeds, texts, progress.tqdm)
    wavs = [_[0] for _ in seeds]
    seeds = [f"保存种子 {_[1]}" for _ in seeds]
    # 不足的部分
    all_wavs = wavs + [None] * (max_audio_components - len(wavs))
    all_seeds = seeds + [''] * (max_audio_components - len(seeds))
    return [item for pair in zip(all_wavs, all_seeds, all_wavs) for item in pair]


# 保存刚刚生成的种子文件路径
audio_paths = [gr.State(value=None) for _ in range(max_audio_components)]


def audio_interface_with_paths(num_seeds, texts, progress=gr.Progress()):
    """
    比 audio_interface 多携带音频的 path
    """
    results = audio_interface(num_seeds, texts, progress)
    wavs = results[::2]  # 提取音频文件路径
    for i, wav in enumerate(wavs):
        audio_paths[i].value = wav  # 直接为 State 组件赋值
    return results


def audio_interface_empty(num_seeds, texts, progress=gr.Progress(track_tqdm=True)):
    return [None, "", None] * max_audio_components


def update_audio_components(slider_value):
    # 根据滑块的值更新 Audio 组件的可见性
    k = int(slider_value)
    audios = [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (max_audio_components - k)
    tbs = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (max_audio_components - k)
    stats = [gr.State(value=None)] * max_audio_components
    print(f'k={k}, audios={len(audios)}')
    return [item for pair in zip(audios, tbs, stats) for item in pair]


def seed_change(evt: gr.SelectData):
    # print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    global SELECTED_SEED_INDEX
    SELECTED_SEED_INDEX = evt.index
    return evt.index


def generate_tts_audio(text_file, num_seeds, seed, speed, oral, laugh, bk, min_length, batch_size, temperature, top_P,
                       top_K, roleid=None, refine_text=True, speaker_type="seed", pt_file=None, progress=gr.Progress()):
    from tts_model import generate_audio_for_seed
    from utils import split_text, replace_tokens, restore_tokens
    if seed in [0, -1, None]:
        seed = random.randint(1, 9999)
    content = ''
    if os.path.isfile(text_file):
        content = ""
    elif isinstance(text_file, str):
        content = text_file
    # 将  [uv_break]  [laugh] 替换为 _uv_break_ _laugh_ 处理后再还原
    content = replace_tokens(content)
    texts = split_text(content, min_length=min_length)
    for i, text in enumerate(texts):
        texts[i] = restore_tokens(text)

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    try:
        output_files = generate_audio_for_seed(
            chat=chat,
            seed=seed,
            texts=texts,
            batch_size=batch_size,
            speed=speed,
            refine_text_prompt=refine_text_prompt,
            roleid=roleid,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            cur_tqdm=progress.tqdm,
            skip_save=False,
            skip_refine_text=not refine_text,
            speaker_type=speaker_type,
            pt_file=pt_file,
        )
        return output_files
    except Exception as e:
        raise e


def generate_refine(text_file, oral, laugh, bk, temperature, top_P, top_K, progress=gr.Progress()):
    from tts_model import generate_refine_text
    from utils import split_text, replace_tokens, restore_tokens, replace_space_between_chinese
    seed = random.randint(1, 9999)
    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    content = ''
    if os.path.isfile(text_file):
        content = ""
    elif isinstance(text_file, str):
        content = text_file
    if re.search(r'\[uv_break\]|\[laugh\]', content) is not None:
        gr.Info("检测到 [uv_break] [laugh]，不能重复 refine ")
        # print("检测到 [uv_break] [laugh]，不能重复 refine ")
        return content
    batch_size = 5

    content = replace_tokens(content)
    texts = split_text(content, min_length=120)
    print(texts)
    for i, text in enumerate(texts):
        texts[i] = restore_tokens(text)
    txts = []
    for batch in progress.tqdm(batch_split(texts, batch_size), desc=f"Refine Text Please Wait ..."):
        txts.extend(generate_refine_text(chat, seed, batch, refine_text_prompt, temperature, top_P, top_K))
    return replace_space_between_chinese('\n\n'.join(txts))


def generate_seed():
    new_seed = random.randint(1, 9999)
    return {
        "__type__": "update",
        "value": new_seed
    }


def update_label(text):
    word_count = len(text)
    return gr.update(label=f"朗读文本（{word_count} 字）")


def inser_token(text, btn):
    if btn == "+笑声":
        return gr.update(
            value=text + "[laugh]"
        )
    elif btn == "+停顿":
        return gr.update(
            value=text + "[uv_break]"
        )


with gr.Blocks() as demo:
    # 项目链接
    gr.Markdown("""
        <div style='text-align: center; font-size: 16px;'>
            🌟  <a href='https://github.com/6drf21e/ChatTTS_colab'>项目地址 欢迎 start</a> 🌟
        </div>
        """)

    with gr.Tab("音色抽卡"):
        with gr.Row():
            with gr.Column(scale=1):
                texts = [
                    "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
                    "我是一个充满活力的人，喜欢运动，喜欢旅行，喜欢尝试新鲜事物。我喜欢挑战自己，不断突破自己的极限，让自己变得更加强大。",
                    "罗森宣布将于7月24日退市，在华门店超6000家！",
                ]
                # gr.Markdown("### 随机音色抽卡")
                gr.Markdown("""
                在相同的 seed 和 温度等参数下，音色具有一定的一致性。点击下面的“随机音色生成”按钮将生成多个 seed。找到满意的音色后，点击音频下方“保存”按钮。
                **注意：不同机器使用相同种子生成的音频音色可能不同，同一机器使用相同种子多次生成的音频音色也可能变化。**
                """)
                input_text = gr.Textbox(label="测试文本",
                                        info="**每行文本**都会生成一段音频，最终输出的音频是将这些音频段合成后的结果。建议使用**多行文本**进行测试，以确保音色稳定性。",
                                        lines=4, placeholder="请输入文本...", value='\n'.join(texts))

                num_seeds = gr.Slider(minimum=1, maximum=max_audio_components, step=1, label="seed生成数量",
                                      value=num_seeds_default)

                generate_button = gr.Button("随机音色抽卡🎲", variant="primary")

                # 保存的种子
                gr.Markdown("### 种子管理界面")
                seed_list = gr.DataFrame(
                    label="种子列表",
                    headers=["Index", "Seed", "Name", "Path"],
                    datatype=["number", "number", "str", "str"],
                    interactive=True,
                    col_count=(4, "fixed"),
                    value=display_seeds
                )

                with gr.Row():
                    refresh_button = gr.Button("刷新")
                    save_button = gr.Button("保存")
                    del_button = gr.Button("删除")
                    play_button = gr.Button("试听")

                with gr.Row():
                    # 添加已保存的种子音频播放组件
                    audio_player = gr.Audio(label="播放已保存种子音频", visible=False)

                # 绑定按钮和函数
                refresh_button.click(display_seeds, outputs=seed_list)
                seed_list.select(seed_change).success(seed_change_btn, outputs=[del_button, play_button])
                save_button.click(do_save_seeds, inputs=[seed_list], outputs=None)
                del_button.click(do_delete_seed, inputs=del_button, outputs=seed_list)
                play_button.click(do_play_seed, inputs=play_button, outputs=audio_player)

            with gr.Column(scale=1):
                audio_components = []
                for i in range(max_audio_components):
                    visible = i < num_seeds_default
                    a = gr.Audio(f"Audio {i}", visible=visible)
                    t = gr.Button(f"Seed", visible=visible)
                    s = gr.State(value=None)
                    t.click(do_save_seed, inputs=[t, s], outputs=None).success(display_seeds, outputs=seed_list)
                    audio_components.append(a)
                    audio_components.append(t)
                    audio_components.append(s)

                num_seeds.change(update_audio_components, inputs=num_seeds, outputs=audio_components)
                # output = gr.Column()
                # audio = gr.Audio(label="Output Audio")

            generate_button.click(
                audio_interface_empty,
                inputs=[num_seeds, input_text],
                outputs=audio_components
            ).success(audio_interface, inputs=[num_seeds, input_text], outputs=audio_components)
    with gr.Tab("长音频生成"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 文本")
                # gr.Markdown("请上传要转换的文本文件（.txt 格式）。")
                # text_file_input = gr.File(label="文本文件", file_types=[".txt"])
                default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
                text_file_input = gr.Textbox(label=f"朗读文本（字数: {len(default_text)}）", lines=4,
                                             placeholder="Please Input Text...", value=default_text)
                # 当文本框内容发生变化时调用 update_label 函数
                text_file_input.change(update_label, inputs=text_file_input, outputs=text_file_input)
                # 加入停顿按钮
                with gr.Row():
                    break_button = gr.Button("+停顿", variant="secondary")
                    laugh_button = gr.Button("+笑声", variant="secondary")
                refine_button = gr.Button("Refine Text（预处理 加入停顿词、笑声等）", variant="secondary")

            with gr.Column():
                gr.Markdown("### 配置参数")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("音色选择")
                        num_seeds_input = gr.Number(label="生成音频的数量", value=1, precision=0, visible=False)
                        speaker_stat = gr.State(value="seed")
                        tab_seed = gr.Tab(label="种子")
                        with tab_seed:
                            with gr.Row():
                                seed_input = gr.Number(label="指定种子", info="种子决定音色 0则随机", value=None,
                                                       precision=0)
                                generate_audio_seed = gr.Button("\U0001F3B2")
                        tab_roleid = gr.Tab(label="内置音色")
                        with tab_roleid:
                            roleid_input = gr.Dropdown(label="内置音色",
                                                       choices=[("发姐", "1"),
                                                                ("纯情男大学生", "2"),
                                                                ("阳光开朗大男孩", "3"),
                                                                ("知心小姐姐", "4"),
                                                                ("电视台女主持", "5"),
                                                                ("魅力大叔", "6"),
                                                                ("优雅甜美", "7"),
                                                                ("贴心男宝2", "21"),
                                                                ("正式打工人", "8"),
                                                                ("贴心男宝1", "9")],
                                                       value="1",
                                                       info="选择音色后会覆盖种子。感谢 @QuantumDriver 提供音色")
                        tab_pt = gr.Tab(label="上传.PT文件")
                        with tab_pt:
                            pt_input = gr.File(label="上传音色文件", file_types=[".pt"], height=100)

                with gr.Row():
                    style_select = gr.Radio(label="预设参数", info="语速部分可自行更改",
                                            choices=["小说朗读", "闲聊", "默认"], interactive=True, )
                with gr.Row():
                    # refine
                    refine_text_input = gr.Checkbox(label="Refine",
                                                    info="打开后会自动根据下方参数添加笑声/停顿等。关闭后可自行添加 [uv_break] [laugh] 或者点击下方 Refin按钮先行转换",
                                                    value=True)
                    speed_input = gr.Slider(label="语速", minimum=1, maximum=10, value=DEFAULT_SPEED, step=1)
                with gr.Row():
                    oral_input = gr.Slider(label="口语化", minimum=0, maximum=9, value=DEFAULT_ORAL, step=1)
                    laugh_input = gr.Slider(label="笑声", minimum=0, maximum=2, value=DEFAULT_LAUGH, step=1)
                    bk_input = gr.Slider(label="停顿", minimum=0, maximum=7, value=DEFAULT_BK, step=1)
                # gr.Markdown("### 文本参数")
                with gr.Row():
                    min_length_input = gr.Number(label="文本分段长度", info="大于这个数值进行分段",
                                                 value=DEFAULT_SEG_LENGTH, precision=0)
                    batch_size_input = gr.Number(label="批大小", info="越高越快 太高爆显存 4G推荐3 其他酌情",
                                                 value=DEFAULT_BATCH_SIZE, precision=0)
                with gr.Accordion("其他参数", open=False):
                    with gr.Row():
                        # 温度 top_P top_K
                        temperature_input = gr.Slider(label="温度", minimum=0.01, maximum=1.0, step=0.01,
                                                      value=DEFAULT_TEMPERATURE)
                        top_P_input = gr.Slider(label="top_P", minimum=0.1, maximum=0.9, step=0.05, value=DEFAULT_TOP_P)
                        top_K_input = gr.Slider(label="top_K", minimum=1, maximum=20, step=1, value=DEFAULT_TOP_K)
                        # reset 按钮
                        reset_button = gr.Button("重置")

        with gr.Row():
            generate_button = gr.Button("生成音频", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(label="生成的音频文件")

        generate_audio_seed.click(generate_seed,
                                  inputs=[],
                                  outputs=seed_input)


        def do_tab_change(evt: gr.SelectData):
            print(evt.selected, evt.index, evt.value, evt.target)
            kv = {
                "种子": "seed",
                "内置音色": "role",
                "上传.PT文件": "pt"
            }
            return kv.get(evt.value, "seed")


        tab_seed.select(do_tab_change, outputs=speaker_stat)
        tab_roleid.select(do_tab_change, outputs=speaker_stat)
        tab_pt.select(do_tab_change, outputs=speaker_stat)


        def do_style_select(x):
            if x == "小说朗读":
                return [4, 0, 0, 2]
            elif x == "闲聊":
                return [5, 5, 1, 4]
            else:
                return [DEFAULT_SPEED, DEFAULT_ORAL, DEFAULT_LAUGH, DEFAULT_BK]


        # style_select 选择
        style_select.change(
            do_style_select,
            inputs=style_select,
            outputs=[speed_input, oral_input, laugh_input, bk_input]
        )

        # refine 按钮
        refine_button.click(
            generate_refine,
            inputs=[text_file_input, oral_input, laugh_input, bk_input, temperature_input, top_P_input, top_K_input],
            outputs=text_file_input
        )
        # 重置按钮 重置温度等参数
        reset_button.click(
            lambda: [0.3, 0.7, 20],
            inputs=None,
            outputs=[temperature_input, top_P_input, top_K_input]
        )

        generate_button.click(
            fn=generate_tts_audio,
            inputs=[
                text_file_input,
                num_seeds_input,
                seed_input,
                speed_input,
                oral_input,
                laugh_input,
                bk_input,
                min_length_input,
                batch_size_input,
                temperature_input,
                top_P_input,
                top_K_input,
                roleid_input,
                refine_text_input,
                speaker_stat,
                pt_input
            ],
            outputs=[output_audio]
        )

        break_button.click(
            inser_token,
            inputs=[text_file_input, break_button],
            outputs=text_file_input
        )

        laugh_button.click(
            inser_token,
            inputs=[text_file_input, laugh_button],
            outputs=text_file_input
        )

    with gr.Tab("角色扮演"):
        def txt_2_script(text):
            lines = text.split("\n")
            data = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("::")
                if len(parts) != 2:
                    continue
                data.append({
                    "character": parts[0],
                    "txt": parts[1]
                })
            return data


        def script_2_txt(data):
            assert isinstance(data, list)
            result = []
            for item in data:
                txt = item['txt'].replace('\n', ' ')
                result.append(f"{item['character']}::{txt}")
            return "\n".join(result)


        def get_characters(lines):
            assert isinstance(lines, list)
            characters = list([_["character"] for _ in lines])
            unique_characters = list(dict.fromkeys(characters))
            print([[character, 0] for character in unique_characters])
            return [[character, 0, 5, 2, 0, 4] for character in unique_characters]


        def get_txt_characters(text):
            return get_characters(txt_2_script(text))


        def llm_change(model):
            llm_setting = {
                "gpt-3.5-turbo-0125": ["https://api.openai.com/v1"],
                "gpt-4o": ["https://api.openai.com/v1"],
                "deepseek-chat": ["https://api.deepseek.com"],
                "yi-large": ["https://api.lingyiwanwu.com/v1"]
            }
            if model in llm_setting:
                return llm_setting[model][0]
            else:
                gr.Error("Model not found.")
                return None


        def ai_script_generate(model, api_base, api_key, text, progress=gr.Progress(track_tqdm=True)):
            from llm_utils import llm_operation
            from config import LLM_PROMPT
            scripts = llm_operation(api_base, api_key, model, LLM_PROMPT, text, required_keys=["txt", "character"])
            return script_2_txt(scripts)


        def generate_script_audio(text, models_seeds, progress=gr.Progress()):
            scripts = txt_2_script(text)  # 将文本转换为剧本
            characters = get_characters(scripts)  # 从剧本中提取角色

            #
            import pandas as pd
            from collections import defaultdict
            import itertools
            from tts_model import generate_audio_for_seed
            from utils import combine_audio, save_audio, normalize_zh

            assert isinstance(models_seeds, pd.DataFrame)

            # 批次处理函数
            def batch(iterable, batch_size):
                it = iter(iterable)
                while True:
                    batch = list(itertools.islice(it, batch_size))
                    if not batch:
                        break
                    yield batch

            column_mapping = {
                '角色': 'character',
                '种子': 'seed',
                '语速': 'speed',
                '口语': 'oral',
                '笑声': 'laugh',
                '停顿': 'break'
            }
            # 使用 rename 方法重命名 DataFrame 的列
            models_seeds = models_seeds.rename(columns=column_mapping).to_dict(orient='records')
            # models_seeds = models_seeds.to_dict(orient='records')

            # 检查每个角色是否都有对应的种子
            print(models_seeds)
            seed_lookup = {seed['character']: seed for seed in models_seeds}

            character_seeds = {}
            missing_seeds = []
            # 遍历所有角色
            for character in characters:
                character_name = character[0]
                seed_info = seed_lookup.get(character_name)
                if seed_info:
                    character_seeds[character_name] = seed_info
                else:
                    missing_seeds.append(character_name)

            if missing_seeds:
                missing_characters_str = ', '.join(missing_seeds)
                gr.Info(f"以下角色没有种子，请先设置种子：{missing_characters_str}")
                return None

            print(character_seeds)
            # return
            refine_text_prompt = "[oral_2][laugh_0][break_4]"
            all_wavs = []

            # 按角色分组，加速推理
            grouped_lines = defaultdict(list)
            for line in scripts:
                grouped_lines[line["character"]].append(line)

            batch_results = {character: [] for character in grouped_lines}

            batch_size = 5  # 设置批次大小
            # 按角色处理
            for character, lines in progress.tqdm(grouped_lines.items(), desc="生成剧本音频"):
                info = character_seeds[character]
                seed = info["seed"]
                speed = info["speed"]
                orla = info["oral"]
                laugh = info["laugh"]
                bk = info["break"]

                refine_text_prompt = f"[oral_{orla}][laugh_{laugh}][break_{bk}]"

                # 按批次处理
                for batch_lines in batch(lines, batch_size):
                    texts = [normalize_zh(line["txt"]) for line in batch_lines]
                    print(f"seed={seed} t={texts} c={character} s={speed} r={refine_text_prompt}")
                    wavs = generate_audio_for_seed(chat, int(seed), texts, DEFAULT_BATCH_SIZE, speed,
                                                   refine_text_prompt, None, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
                                                   DEFAULT_TOP_K, skip_save=True)  # 批量处理文本
                    batch_results[character].extend(wavs)

            # 转换回原排序
            for line in scripts:
                character = line["character"]
                all_wavs.append(batch_results[character].pop(0))

            # 合成所有音频
            audio = combine_audio(all_wavs)
            fname = f"script_{int(time.time())}.wav"
            return save_audio(fname, audio)


        script_example = {
            "lines": [{
                "txt": "在一个风和日丽的下午，小红帽准备去森林里看望她的奶奶。",
                "character": "旁白"
            }, {
                "txt": "小红帽说",
                "character": "旁白"
            }, {
                "txt": "我要给奶奶带点好吃的。",
                "character": "年轻女性"
            }, {
                "txt": "在森林里，小红帽遇到了狡猾的大灰狼。",
                "character": "旁白"
            }, {
                "txt": "大灰狼说",
                "character": "旁白"
            }, {
                "txt": "小红帽，你的篮子里装的是什么？",
                "character": "中年男性"
            }, {
                "txt": "小红帽回答",
                "character": "旁白"
            }, {
                "txt": "这是给奶奶的蛋糕和果酱。",
                "character": "年轻女性"
            }, {
                "txt": "大灰狼心生一计，决定先到奶奶家等待小红帽。",
                "character": "旁白"
            }, {
                "txt": "当小红帽到达奶奶家时，她发现大灰狼伪装成了奶奶。",
                "character": "旁白"
            }, {
                "txt": "小红帽疑惑的问",
                "character": "旁白"
            }, {
                "txt": "奶奶，你的耳朵怎么这么尖？",
                "character": "年轻女性"
            }, {
                "txt": "大灰狼慌张地回答",
                "character": "旁白"
            }, {
                "txt": "哦，这是为了更好地听你说话。",
                "character": "中年男性"
            }, {
                "txt": "小红帽越发觉得不对劲，最终发现了大灰狼的诡计。",
                "character": "旁白"
            }, {
                "txt": "她大声呼救，森林里的猎人听到后赶来救了她和奶奶。",
                "character": "旁白"
            }, {
                "txt": "从此，小红帽再也没有单独进入森林，而是和家人一起去看望奶奶。",
                "character": "旁白"
            }]
        }

        ai_text_default = "武侠小说《花木兰大战周树人》 要符合人物背景"

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### AI脚本")
                gr.Markdown("""
为确保生成效果稳定，仅支持与 GPT-4 相当的模型，推荐使用 4o yi-large deepseek。
如果没有反应，请检查日志中的错误信息。如果提示格式错误，请重试几次。国内模型可能会受到风控影响，建议更换文本内容后再试。

申请渠道（免费额度）：

- [https://platform.deepseek.com/](https://platform.deepseek.com/)
- [https://platform.lingyiwanwu.com/](https://platform.lingyiwanwu.com/)

                """)
                # 申请渠道

                with gr.Row(equal_height=True):
                    # 选择模型 只有 gpt4o deepseek-chat yi-large 三个选项
                    model_select = gr.Radio(label="选择模型", choices=["gpt-4o", "deepseek-chat", "yi-large"],
                                            value="gpt-4o", interactive=True, )
                with gr.Row(equal_height=True):
                    openai_api_base_input = gr.Textbox(label="OpenAI API Base URL",
                                                       placeholder="请输入API Base URL",
                                                       value=r"https://api.openai.com/v1")
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="请输入API Key",
                                                      value="sk-xxxxxxx", type="password")
                # AI提示词
                ai_text_input = gr.Textbox(label="剧情简介或者一段故事", placeholder="请输入文本...", lines=2,
                                           value=ai_text_default)

                # 生成脚本的按钮
                ai_script_generate_button = gr.Button("AI脚本生成")

            with gr.Column(scale=3):
                gr.Markdown("### 脚本")
                gr.Markdown(
                    "脚本可以手工编写也可以从左侧的AI脚本生成按钮生成。脚本格式 **角色::文本** 一行为一句” 注意是::")
                script_text = "\n".join(
                    [f"{_.get('character', '')}::{_.get('txt', '')}" for _ in script_example['lines']])

                script_text_input = gr.Textbox(label="脚本格式 “角色::文本 一行为一句” 注意是::",
                                               placeholder="请输入文本...",
                                               lines=12, value=script_text)
                script_translate_button = gr.Button("步骤①：提取角色")

            with gr.Column(scale=1):
                gr.Markdown("### 角色种子")
                # DataFrame 来存放转换后的脚本
                # 默认数据 [speed_5][oral_2][laugh_0][break_4]
                default_data = [
                    ["旁白", 2222, 3, 0, 0, 2],
                    ["年轻女性", 2, 5, 2, 0, 2],
                    ["中年男性", 2424, 5, 2, 0, 2]
                ]

                script_data = gr.DataFrame(
                    value=default_data,
                    label="角色对应的音色种子，从抽卡那获取",
                    headers=["角色", "种子", "语速", "口语", "笑声", "停顿"],
                    datatype=["str", "number", "number", "number", "number", "number"],
                    interactive=True,
                    col_count=(6, "fixed"),
                )
                # 生视频按钮
                script_generate_audio = gr.Button("步骤②：生成音频")
        # 输出的脚本音频
        script_audio = gr.Audio(label="AI生成的音频", interactive=False)

        # 脚本相关事件
        # 脚本转换
        script_translate_button.click(
            get_txt_characters,
            inputs=[script_text_input],
            outputs=script_data
        )
        # 处理模型切换
        model_select.change(
            llm_change,
            inputs=[model_select],
            outputs=[openai_api_base_input]
        )
        # AI脚本生成
        ai_script_generate_button.click(
            ai_script_generate,
            inputs=[model_select, openai_api_base_input, openai_api_key_input, ai_text_input],
            outputs=[script_text_input]
        )
        # 音频生成
        script_generate_audio.click(
            generate_script_audio,
            inputs=[script_text_input, script_data],
            outputs=[script_audio]
        )

demo.launch(share=args.share, inbrowser=True)
