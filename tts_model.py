import ChatTTS
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import datetime
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


def generate_audio_for_seed(chat, seed, texts, batch_size, speed, refine_text_prompt, temperature=DEFAULT_TEMPERATURE,
                            top_P=DEFAULT_TOP_P, top_K=DEFAULT_TOP_K, cur_tqdm=None, skip_save=False):
    from utils import combine_audio, save_audio, batch_split
    # torch.manual_seed(seed)
    # top_P = 0.7,
    # top_K = 20,
    # temperature = 0.3,
    if seed in [None, -1, 0, "", "random"]:
        seed = np.random.randint(0, 9999)

    deterministic(seed)
    rnd_spk_emb = chat.sample_random_speaker()
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

    for batch in cur_tqdm(batch_split(texts, batch_size), desc=f"Inferring audio for seed={seed}"):
        flag += len(batch)
        # refine_text =  chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text, refine_text_only=True)
        # print(refine_text)
        # exit()
        wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text,
                          use_decoder=True, skip_refine_text=False)
        all_wavs.extend(wavs)
        clear_cuda_cache()
    if skip_save:
        return all_wavs
    combined_audio = combine_audio(all_wavs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Saving audio for seed {seed}, took {elapsed_time:.2f}s")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wav_filename = f"long-[seed_{seed}][speed_{speed}]{refine_text_prompt}[{timestamp}].wav"
    save_audio(wav_filename, combined_audio)
    return wav_filename


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
