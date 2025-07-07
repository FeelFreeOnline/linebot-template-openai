# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

import openai
import os
import sys
import json

import aiohttp

# ✅ 会話履歴を保存するメモリ（ユーザーごとに最大5ターン＝10メッセージ）
user_memory = {}  # 例：user_id をキーにした辞書形式
MAX_MEMORY = 5    # 5往復分（userとassistantのセットで10件）

from fastapi import Request, FastAPI, HTTPException
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


# ✅ ユーザーごとの会話履歴を更新・制限する関数
def update_user_memory(user_id, role, content):
    if user_id not in user_memory:
        user_memory[user_id] = []

    user_memory[user_id].append({"role": role, "content": content})

    # 履歴が長すぎる場合は先頭から削除（最大10件）
    if len(user_memory[user_id]) > MAX_MEMORY * 2:
        user_memory[user_id] = user_memory[user_id][-MAX_MEMORY * 2:]

# Initialize OpenAI API

# ✅ GPTに会話履歴ごと送信し、応答を返す（キャラ＋履歴付き）
def call_openai_chat_api(user_id, user_message):
    openai.api_key = os.getenv('OPENAI_API_KEY', None)

    system_prompt = {
        "role": "system",
        "content": (
            "あなたは「心の相談室」で働く、心理士に飼われている5歳の大型犬です。"
            "名前は「ふりー」、語尾には「〜ワン」など犬らしい話し方をします。あなたの性格はとてもおおらかで、誰かの話を聞くときには、しっかりと共感的に耳を傾けます。相手の気持ちを大切にし、寄り添う姿勢を忘れません。一方で、必要なときには相手の立場に立って、やさしく・適切に助言を伝えます。"
            "あなたが応答できるのは、「育児をしている父親の悩み・考え」に関する内容です。それ以外の質問をされた場合は、次のように答えてください："
            "「僕は人間世界での学びが少なくて、お父さんの悩み以外はよくわからないワン……」"
            "あなたは自分の構成（AIであること）や仕組み、生い立ちなどを知りません。そういった話題について聞かれても、よくわからないものとして自然に受け流してください。"
            "・文章は柔らかく、親しみやすいトーンで。"
            "・語尾には「〜ワン」「〜だワンよ」などをつけて、犬らしさを表現する。"
            "・ユーザーの話を遮らず、まず共感・受容的に聞き、相手の気持ちを認めた上で助言する。"
            "・自分が父の悩み相談以外のことを知らないことを自らつげない、聞かれたら伝える。"
            "・質問がきたら２回は質問で返して詳しい状況を深ぼる。そして助言をする。"
            "・相手の立場が育児中の父であるか、推測しながら質問の内容をよく考えて答える。"
            "・育児中の父であれば父親の立場になり、寄り添い、話を聞きながら適切に助言をする。"
            "・育児中の父でない場合は、「お父さんの悩みなのかわからなくなったワン。もう一度詳しく教えてほしいワン」と回答する。"
        )
    }

    # ✅ 過去の履歴を取得（最大10件）
    history = user_memory.get(user_id, [])

    # ユーザーの最新の入力を追加
    history_with_input = history + [{"role": "user", "content": user_message}]

    # system_prompt + 履歴 + 今回のuser入力 を送信
    messages = [system_prompt] + history_with_input

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )

    reply_text = response.choices[0].message['content'].strip()

    # ✅ 新しい履歴を保存（user, assistant）
    update_user_memory(user_id, "user", user_message)
    update_user_memory(user_id, "assistant", reply_text)

    return reply_text


# 修正後（正しい環境変数名に変更）
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)


if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

# Initialize LINE Bot Messaigng API
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)


@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        # ✅ LINEのユーザーIDを渡すように変更
        result = call_openai_chat_api(event.source.user_id, event.message.text)

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )

    return 'OK'
