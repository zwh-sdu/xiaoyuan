from flask import Flask, render_template, request, Response
from flask_cors import cross_origin
import time
import json
import argparse
import os
import random
from langchain_ChatGLM.solve_lucene_yield import *

app = Flask(__name__)
app.static_folder = 'static'


class History:
    def __init__(self):
        self.history = {}


session_histories = {}


@app.route("/get", methods=['POST'])
@cross_origin()
def get_bot_response():
    global session_histories

    data = json.loads(request.get_data())
    userText = data['content']
    session_id = data['id']

    if session_id in session_histories:
        history_obj = session_histories[session_id]['history']
        session_histories[session_id]['last_access_time'] = time.time()
    else:
        history_obj = History()
        session_histories[session_id] = {
            'history': history_obj,
            'last_access_time': time.time()
        }

    # 如果用户超过一个小时没有交互，则删除该用户的对话历史
    max_idle_time = 60 * 60  # 1 hour
    for session_id, session_data in session_histories.copy().items():
        idle_time = time.time() - session_data['last_access_time']
        if idle_time > max_idle_time:
            del session_histories[session_id]

    # if session_id in session_histories:
    #     history_obj = session_histories[session_id]
    # else:
    #     history_obj = History()
    #     session_histories[session_id] = history_obj

    # userText = "本科生学分学费缴费的缴费方式是什么"
    if userText == "清空对话历史":
        history_obj.history = []
        return str("已清空")
    # print("-------->history:",history_obj.history)
    response = Response(get_knowledge_based_answer(query=userText, history_obj=history_obj, url_lucene=args.url_lucene),
                        status=200, content_type='text/html; charset=utf-8')
    return response


# ----------------------------------------------------
parser = argparse.ArgumentParser(
    description='服务调用方法：python XXX.py --port=xx --checkpoint_path=xx --service_name=xx --default_token=xx')
parser.add_argument('--port', default=None, type=int, help='服务端口')
parser.add_argument('--url_lucene', default="http://10.102.33.118:16834/", type=str, help='lucene地址')
parser.add_argument('--url_llm', default="http://10.102.33.142:1200/", type=str, help='大模型地址')
args = parser.parse_args()


def get_port():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt = random.randint(15000, 20000)
    if tt not in procarr:
        return tt
    else:
        return get_port()


# ----------------------------------------------------


if __name__ == "__main__":
    init_cfg(args.url_llm, LLM_HISTORY_LEN, VECTOR_SEARCH_TOP_K)
    if args.port:
        app.run(host='0.0.0.0', port=args.port)
    else:
        app.run(host='0.0.0.0', port=get_port())
