from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# DEVICE = "cuda"
# DEVICE_ID = "0"
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# def torch_gc():
#     if torch.cuda.is_available():
#         with torch.cuda.device(CUDA_DEVICE):
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()

app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('input')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history if history else [],
                                   max_length=max_length if max_length else 4096,
                                   max_time=2,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    # print(log)
    # torch_gc()
    return answer


if __name__ == '__main__':
    llm_path = "/data03/irlab_share/chatglm"
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(llm_path, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=1111, workers=1)
