import datetime
import json
import os
import re
import time

import numpy as np
import torch
from tqdm import tqdm

import ChatTTS
from config import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K


def load_chat_tts_model(source='huggingface', force_redownload=False, local_path=None):
    """
    Load ChatTTS model
    :param source:
    :param force_redownload:
    :param local_path:
    :return:
    """
    print("Loading ChatTTS model...")
    chat = ChatTTS.Chat()
    chat.load_models(source=source, force_redownload=force_redownload, local_path=local_path)
    return chat


def clear_cuda_cache():
    """
    Clear CUDA cache
    :return:
    """
    torch.cuda.empty_cache()


def deterministic(seed=0):
    """
    Set random seed for reproducibility
    :param seed:
    :return:
    """
    # ref: https://github.com/Jackiexiao/ChatTTS-api-ui-docker/blob/main/api.py#L27
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_audio_for_seed(chat, seed, texts, batch_size, speed, refine_text_prompt, roleid=None,
                            temperature=DEFAULT_TEMPERATURE,
                            top_P=DEFAULT_TOP_P, top_K=DEFAULT_TOP_K, cur_tqdm=None, skip_save=False,
                            skip_refine_text=False, speaker_type="seed", pt_file=None):
    from utils import combine_audio, save_audio, batch_split
    print(f"speaker_type: {speaker_type}")
    if speaker_type == "seed":
        if seed in [None, -1, 0, "", "random"]:
            seed = np.random.randint(0, 9999)
        deterministic(seed)
        rnd_spk_emb = chat.sample_random_speaker()
    elif speaker_type == "role":
        # 从 JSON 文件中读取数据
        with open('./slct_voice_240605.json', 'r', encoding='utf-8') as json_file:
            slct_idx_loaded = json.load(json_file)
        # 将包含 Tensor 数据的部分转换回 Tensor 对象
        for key in slct_idx_loaded:
            tensor_list = slct_idx_loaded[key]["tensor"]
            slct_idx_loaded[key]["tensor"] = torch.tensor(tensor_list)
        # 将音色 tensor 打包进params_infer_code，固定使用此音色发音，调低temperature
        rnd_spk_emb = slct_idx_loaded[roleid]["tensor"]
        # temperature = 0.001
    elif speaker_type == "pt":
        print(pt_file)
        rnd_spk_emb = torch.load(pt_file)
        print(rnd_spk_emb.shape)
        if rnd_spk_emb.shape != (768,):
            raise ValueError("维度应为 768。")
    else:
        raise ValueError(f"Invalid speaker_type: {speaker_type}. ")

    params_infer_code = {
        'spk_emb': rnd_spk_emb,
        'prompt': f'[speed_{speed}]',
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature
    }
    params_refine_text = {
        'prompt': refine_text_prompt,
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature
    }
    all_wavs = []
    start_time = time.time()
    total = len(texts)
    flag = 0
    if not cur_tqdm:
        cur_tqdm = tqdm

    if re.search(r'\[uv_break\]|\[laugh\]', ''.join(texts)) is not None:
        if not skip_refine_text:
            print("Detected [uv_break] or [laugh] in text, skipping refine_text")
        skip_refine_text = True

    for batch in cur_tqdm(batch_split(texts, batch_size), desc=f"Inferring audio for seed={seed}"):
        flag += len(batch)
        wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text,
                          use_decoder=True, skip_refine_text=skip_refine_text)
        all_wavs.extend(wavs)
        clear_cuda_cache()
    if skip_save:
        return all_wavs
    combined_audio = combine_audio(all_wavs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Saving audio for seed {seed}, took {elapsed_time:.2f}s")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wav_filename = f"chattts-[seed_{seed}][speed_{speed}]{refine_text_prompt}[{timestamp}].wav"
    return save_audio(wav_filename, combined_audio)


def generate_refine_text(chat, seed, text, refine_text_prompt, temperature=DEFAULT_TEMPERATURE,
                         top_P=DEFAULT_TOP_P, top_K=DEFAULT_TOP_K):
    if seed in [None, -1, 0, "", "random"]:
        seed = np.random.randint(0, 9999)

    deterministic(seed)

    params_refine_text = {
        'prompt': refine_text_prompt,
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature
    }
    print('params_refine_text:', text)
    print('refine_text_prompt:', refine_text_prompt)
    refine_text = chat.infer(text, params_refine_text=params_refine_text, refine_text_only=True, skip_refine_text=False)
    print('refine_text:', refine_text)
    return refine_text


def tts(chat, text_file, seed, speed, oral, laugh, bk, seg, batch, progres=None):
    """
    Text-to-Speech
    :param chat:  ChatTTS model
    :param text_file:  Text file or string
    :param seed:  Seed
    :param speed:   Speed
    :param oral:  Oral
    :param laugh:  Laugh
    :param bk:
    :param seg:
    :param batch:
    :param progres:
    :return:
    """
    from utils import read_long_text, split_text

    if os.path.isfile(text_file):
        content = read_long_text(text_file)
    elif isinstance(text_file, str):
        content = text_file
    texts = split_text(content, min_length=seg)

    print(texts)
    # exit()

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    return generate_audio_for_seed(chat, seed, texts, batch, speed, refine_text_prompt)
