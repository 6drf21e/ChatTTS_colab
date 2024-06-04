import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import re
import time
from io import BytesIO
import pandas
import numpy as np
from tqdm import tqdm
import random
import os
import gradio as gr
import json
from utils import combine_audio, save_audio, batch_split, normalize_zh
from tts_model import load_chat_tts_model, clear_cuda_cache, deterministic, generate_audio_for_seed
import soundfile as sf
import wave

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse

from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

from pydantic import BaseModel

import uvicorn

from typing import Generator


parser = argparse.ArgumentParser(description="Gradio ChatTTS MIX")
parser.add_argument("--source", type=str, default="local", help="Model source: 'huggingface' or 'local'.")
parser.add_argument("--local_path", type=str,default="models", help="Path to local model if source is 'local'.")
parser.add_argument("--share", default=False, action="store_true", help="Share the server publicly.")

args = parser.parse_args()


class TTS_Request(BaseModel):
    text: str = None
    seed: int = 2581
    speed: int = 3
    media_type: str = "wav"
    streaming: int = 0


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

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。


def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    # opt = "\n".join(mergeitems)
    return mergeitems

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()



### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def generate_tts_audio(text_file,seed=2581,speed=3, oral=0, laugh=0, bk=4, min_length=10, batch_size=10, temperature=0.3, top_P=0.7,
                       top_K=20,streaming=0):

    from tts_model import generate_audio_for_seed
    from utils import split_text

    cur_tqdm = False

    if seed in [0, -1, None]:
        seed = random.randint(1, 9999)
    content = ''
    if os.path.isfile(text_file):
        content = ""
    elif isinstance(text_file, str):
        content = text_file
    texts = split_text(content, min_length=min_length)
    

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"


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

    if not streaming:

        for batch in cur_tqdm(batch_split(texts, batch_size), desc=f"Inferring audio for seed={seed}"):
            flag += len(batch)
            # refine_text =  chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text, refine_text_only=True)
            # print(refine_text)
            # exit()
            wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text,use_decoder=True, skip_refine_text=False)
            audio_data = wavs[0][0]
            audio_data = audio_data / np.max(np.abs(audio_data))


            all_wavs.append(audio_data)

    
            clear_cuda_cache()

        audio = (np.concatenate(all_wavs, 0) * 32768).astype(
                np.int16
            )

        print(audio)

        yield audio

    else:

        print("流式生成")

        texts = cut5(content)

        for text in texts:

            print(text)

            wavs = chat.infer(text, params_infer_code=params_infer_code, params_refine_text=params_refine_text,use_decoder=True, skip_refine_text=False)
            audio_data = wavs[0][0]
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = (audio_data * 32767).astype(np.int16)
            yield audio_data

            clear_cuda_cache()

            



async def tts_handle(req:dict):

    media_type = req["media_type"]

    if not req["streaming"]:
    
        audio_data = next(generate_tts_audio(req["text"],req["seed"]))
        sr = 24000

        print(audio_data)


        audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()


        return Response(audio_data, media_type=f"audio/{media_type}")
    
    else:
        
        tts_generator = generate_tts_audio(req["text"],req["seed"],streaming=1)

        sr = 24000

        def streaming_generator(tts_generator:Generator, media_type:str):
            if media_type == "wav":
                yield wave_header_chunk()
                media_type = "raw"
            for chunk in tts_generator:
                yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

        return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")


@app.get("/")
async def tts_get_endpoint(
                        text: str = None,media_type:str = "wav",seed:int = 2581,streaming:int = 0,
                        ):
    req = {
        "text": text,
        "media_type": media_type,
        "seed": seed,
        "streaming": streaming,
    }
    return await tts_handle(req)


@app.get("/speakers")
def speakers_endpoint():
    return JSONResponse([{"name":"default","vid":1}], status_code=200)


@app.get("/speakers_list")
def speakerlist_endpoint():
    return JSONResponse(["female_calm","female","male"], status_code=200)


@app.post("/")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@app.post("/tts_to_audio/")
async def tts_to_audio(request: TTS_Request):
    req = request.dict()
    from config import llama_seed

    req["seed"] = llama_seed

    return await tts_handle(req)


uvicorn.run(app, host="0.0.0.0", port=9880,workers=1)
