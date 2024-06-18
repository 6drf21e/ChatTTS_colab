import os
import sys
sys.path.insert(0, os.getcwd())
import ChatTTS
import re
import time
import io
from io import BytesIO
import pandas
import numpy as np
from tqdm import tqdm
import random
import os
import json
from utils import batch_split
import torch
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



chat = ChatTTS.Chat()
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


class TTS_Request(BaseModel):
    text: str = None
    seed: int = 2581
    speed: int = 3
    media_type: str = "wav"
    streaming: int = 0






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


def generate_tts_audio(text_file,seed=2,speed=3, oral=0, laugh=0, bk=4, min_length=80, batch_size=5, temperature=0.3, top_P=0.7,
                       top_K=20,streaming=0,cur_tqdm=None):

    from utils import combine_audio, save_audio, batch_split

    from utils import split_text, replace_tokens, restore_tokens


    if seed in [0, -1, None]:
        seed = random.randint(1, 9999)

    
    content = text_file

    # 将  [uv_break]  [laugh] 替换为 _uv_break_ _laugh_ 处理后再还原
    content = replace_tokens(content)
    texts = split_text(content, min_length=min_length)
    for i, text in enumerate(texts):
        texts[i] = restore_tokens(text)

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
    


    if not cur_tqdm:
        cur_tqdm = tqdm

    start_time = time.time()

    if not streaming:

        all_wavs = []



        for batch in cur_tqdm(batch_split(texts, batch_size), desc=f"Inferring audio for seed={seed}"):

            
            wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text,use_decoder=True, skip_refine_text=False)
            audio_data = wavs[0][0]
            audio_data = audio_data / np.max(np.abs(audio_data))


            all_wavs.append(audio_data)

            # all_wavs.extend(wavs)

            clear_cuda_cache()

        

        audio = (np.concatenate(all_wavs) * 32768).astype(
                np.int16
            )

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Saving audio for seed {seed}, took {elapsed_time:.2f}s")

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

            clear_cuda_cache()

            yield audio_data


            



async def tts_handle(req:dict):

    media_type = req["media_type"]

    print(req["streaming"])

    if not req["streaming"]:
    
        audio_data = next(generate_tts_audio(req["text"],req["seed"]))

        # print(audio_data)

        sr = 24000

        audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()


        return Response(audio_data, media_type=f"audio/{media_type}")

        
        # return FileResponse(f"./{audio_data}", media_type="audio/wav")
    
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
async def tts_get(text: str = None,media_type:str = "wav",seed:int = 2581,streaming:int = 0):
    req = {
        "text": text,
        "media_type": media_type,
        "seed": seed,
        "streaming": streaming,
    }
    print("第一次")
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

if __name__ == "__main__":

    chat.load_models(source="local", local_path="models")

    # chat = load_chat_tts_model(source="local", local_path="models")

    uvicorn.run(app,host='0.0.0.0',port=9880,workers=1)
