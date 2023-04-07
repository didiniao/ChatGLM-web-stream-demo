#! /usr/bin/python3
# -*- coding: utf-8 -*-
# ChatGLM-web-stream-demo
# Copyright (c) 2023 TylunasLi, MIT License

from gevent import monkey, pywsgi
monkey.patch_all()
from flask import Flask, Response, request
#from chatglm_service_flask import Flask, request, Response
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import logging
import os
import json
import sys
import time
import hashlib
import base64
import pprint

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = getLogger('ChatGLM', 'chatlog.log')

MAX_HISTORY = 8
custom_charset = b'-_'

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg

model_path = "THUDM/chatglm-6b-int4"

class ChatGLM():
    def __init__(self) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).half().cuda()
        self.model.eval()
        _, _ = self.model.chat(self.tokenizer, "你好", history=[])
        logger.info("Model initialization finished.")
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def answer(self, query: str, history):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history, max_tokens):
        response_time = int(time.time())
        # 计算 SHA-256 哈希值
        hash_obj = hashlib.sha256(response_time.to_bytes(4, byteorder='big'))
        hash_str = hash_obj.digest()

        # 对哈希值进行 base64 编码，得到固定长度的字符串
        encoded_data = base64.b64encode(hash_str, altchars=custom_charset)

        # 将编码后的二进制数据转换为字符串类型
        output_str = encoded_data.decode('ascii')[:30]
        #output_str = base64.b64encode(hash_str).decode('utf-8')[:30]
        response_id = "chatcmpl-" + output_str
    
        if query is None or history is None:
            start_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "model": "glm-6b",
                "created": response_time,
                "choices": [
                    {
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None,
                        "index": 0}
                    ],
                }
            yield start_data
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query, history, max_tokens):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)

            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "model": "glm-6b",
                "created": response_time,
                "choices": [
                    {
                        "delta": {
                            "content": this_response,
                        },
                        "finish_reason": None,
                        "index": 0
                    }
                ],                        
            }      
            yield data
        logger.info("Answer - {}".format(response))
        data_end = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": "glm-6b",
            "created": response_time,
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
        }
        yield data_end
        yield "[DONE]"


def start_server(quantize_level, http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = ChatGLM()
    
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)
    
    @app.route("/")
    def index():
        return Response(json.dumps({'message': 'started', 'success': True}, ensure_ascii=False), content_type="application/json")

    @app.route("/chat", methods=["GET", "POST"])
    def answer_question():
        result = {"query": "", "response": "", "success": False}
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()
                text = arg_dict["query"]
                ori_history = arg_dict["history"]
                logger.info("Query - {}".format(text))
                if len(ori_history) > 0:
                    logger.info("History - {}".format(ori_history))
                history = ori_history[-MAX_HISTORY:]
                history = [tuple(h) for h in history]
                response, history = bot.answer(text, history)
                logger.info("Answer - {}".format(response))
                ori_history.append((text, response))
                result = {"query": text, "response": response,
                          "history": ori_history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")
    
    @app.post("/v1/moderations")
    async def moderations():
        return json.dumps('')
    
    @app.route("/v1/chat/completions", methods=["POST"])
    def answer_question_stream():
        def decorate(generator):
            for item in generator:
                yield format_sse(json.dumps(item, ensure_ascii=False))

        result = {"query": "", "response": "", "success": False}
        text, history = None, None
        ori_history = []
        max_tokens = 2048
        try:
            if "application/json" in request.content_type:
                arg_dict = request.get_json()

                #pprint.pprint(arg_dict)
                try:
                    strem = arg_dict["stream"]
                except:
                    strem = None
                if not strem:
                    result = {"query": "", "response": "", "success": False}
                    try:
                        arg_dict = request.get_json()
                        messages = arg_dict["messages"]
                        for message in messages:
                            content = message['content']

                        for i, msg in enumerate(messages):
                            if msg['role'] == 'user' and i+1 < len(messages):
                                next_msg = messages[i+1]
                                if next_msg['role'] == 'assistant':
                                    ori_history.append((msg['content'], next_msg['content']))

                        if len(ori_history) > 0:
                            logger.info("History - {}".format(ori_history))
                        history = ori_history[-MAX_HISTORY:]
                        history = [tuple(h) for h in history]
                        response, history = bot.answer(content, history)
                        logger.info("Answer - {}".format(response))
                        ori_history.append((content, response))

                        response_time = int(time.time())
                        result = {"id": "chatcmpl-123","object": "chat.completion","created": response_time,
                         "choices": [{"index": 0,
                                      "message": {"role": "assistant",
                                                  "content": response,},
                                                  "finish_reason": "stop"}]}

                    except Exception as e:
                        logger.error(f"error: {e}")
                    return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")

                messages = arg_dict["messages"]
                try:
                    temperature = arg_dict["temperature"]
                except:
                    temperature = 1.0
                try:
                    max_tokens = arg_dict["max_tokens"]
                except:
                    max_tokens = 2048

                for i, msg in enumerate(messages):
                    if msg['role'] == 'user' and i+1 < len(messages):
                        next_msg = messages[i+1]
                        if next_msg['role'] == 'assistant':
                            ori_history.append((msg['content'], next_msg['content']))
                for message in messages:
                    content = message['content']
                logger.info("Query - {}".format(content))
                if len(ori_history) > 0:
                    logger.info("History - {}".format(ori_history))
                history = ori_history[-MAX_HISTORY:]
                history = [tuple(h) for h in history]
        except Exception as e:
            logger.error(f"error 1 : {e}")
        return Response(decorate(bot.stream(content, history, max_tokens)), mimetype='text/event-stream')

    @app.route("/clear", methods=["GET", "POST"])
    def clear():
        history = []
        try:
            bot.clear()
            return Response(json.dumps({"success": True}, ensure_ascii=False), content_type="application/json")
        except Exception as e:
            return Response(json.dumps({"success": False}, ensure_ascii=False), content_type="application/json")

    @app.route("/score", methods=["GET"])
    def score_answer():
        score = request.get("score")
        logger.info("score: {}".format(score))
        return {'success': True}

    logger.info("starting server...")
    server = pywsgi.WSGIServer((http_address, port), app)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for ChatGLM-6B')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--quantize', '-q', help='level of quantize, option：16, 8 or 4', default=16)
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8800)
    args = parser.parse_args()
    start_server(args.quantize, args.host, int(args.port), args.device)
