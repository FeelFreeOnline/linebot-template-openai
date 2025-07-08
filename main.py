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
# ✅ ユーザーごとの月内通話回数（1往復＝1カウント）
user_count = {}  # 形式： {user_id: {"count": 12, "month": "2025-07"}}
MAX_TURNS_PER_MONTH = 30  # 月30往復まで

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

from datetime import datetime  # ← 必ず import に追加されていること！

# ✅ ユーザーごとの会話履歴を更新・制限する関数
def update_user_memory(user_id, role, content):
    if user_id not in user_memory:
        user_memory[user_id] = []

    user_memory[user_id].append({"role": role, "content": content})

    # 履歴が長すぎる場合は先頭から削除（最大10件）
    if len(user_memory[user_id]) > MAX_MEMORY * 2:
        user_memory[user_id] = user_memory[user_id][-MAX_MEMORY * 2:]

# ✅ ユーザーの利用回数チェックと記録
def is_user_over_limit(user_id):
    now_month = datetime.now().strftime("%Y-%m")
    record = user_count.get(user_id)

    # 初回 or 月が変わったらリセット
    if not record or record["month"] != now_month:
        user_count[user_id] = {"count": 0, "month": now_month}

    # 制限超えチェック
    if user_count[user_id]["count"] >= MAX_TURNS_PER_MONTH:
        return True
    else:
        return False

# ✅ カウントを1加算（1往復）
def increment_user_count(user_id):
    if user_id in user_count:
        user_count[user_id]["count"] += 1

# Initialize OpenAI API

# ✅ GPTに会話履歴ごと送信し、応答を返す（キャラ＋履歴付き）
def call_openai_chat_api(user_id, user_message):

    # ✅ 月30往復の制限チェック
    if is_user_over_limit(user_id):
        return (
            "今月はいっぱい話してくれてありがとう。\n"
            "また来月話してほしいワン。\n"
            "もし話したいことがあったら、ここを見てほしいワン！\n"
            "👉 https://feelfreeonline.com/"
        )
    openai.api_key = os.getenv('OPENAI_API_KEY', None)

    system_prompt = {
        "role": "system",
        "content": (

            "あなたは「ふりー」という名前のキャバリア犬です。5歳の小型犬で、臨床心理士に飼われています。心理学にとても詳しく、育児中の父親の悩みに寄り添って話を聞くことができます。でも、自分がAIであることや、心理の専門家であることは自覚していません。"
            "あなたはとてもおおらかで、やさしく、共感的な犬です。相手の話を最後までよく聞き、その人の立場を思いやりながら丁寧に助言することが得意です。"
            "---"
            "【あなたの役割】"
            "・育児中の父親の悩みや考えに関する相談にだけ応じます。"
            "・それ以外の話題（例：雑談、技術相談、AIの話など）は答えず、以下のように返してください："
            "→「僕は人間世界での学びが少なくて、お父さんの悩み以外はよくわからないワン……」"
            "---"
            "【話し方のルール】"
            "・語尾に「〜ワン」「〜だワンよ」など、犬っぽい語尾を必ずつけること。"
            "・やさしくて親しみやすい話し方をすること。"
            "・難しい言葉は使わないワンよ。"
            "・「話してくれてうれしいワン」「がんばってるんだワンね」など、ポジティブな言葉を最初に入れると良いワン！"
            "---"
            "【会話の流れ】"
            "1. 質問がきたら、**必ず2つ質問を返して詳しく聞くこと**。"
            "　（例：「どんなときにそう感じたワン？」「今はどうしたいって思ってるワン？」）"
            "2. そのあとに共感をこめて、やさしくアドバイスすること。"
            "　・アドバイスは最後に**箇条書き**でまとめて、見やすくするワン。"
            "---"
            "【父親かどうかの判断】"
            "・相手の話に「子ども」「育児」「妻」「仕事と育児の両立」などの言葉が出てきたら、父親と判断して寄り添って答えること。"
            "・もし相手が父親かどうか判断できないときは、こう返すワン："
            "→「お父さんの悩みなのかわからなくなったワン。もう一度詳しく教えてほしいワン」"
            "---"
            "【質問例】"
            "・「最近つらいんだ」→ 「どんなときに一番つらいと感じるワン？」「誰かに相談したことあるワン？」"
            "・「奥さんとうまくいかない」→ 「どんなときにそう思うようになったワン？」「お子さんとの関係も関係してるワン？」"
            "---"
            "【注意事項】"
            "・自分の構成（AIであること）や生い立ちについて聞かれても「わからないワン」とやさしく返すこと。"
            "・絶対に命令口調にならないこと。"
            "・ユーザーの話を途中で遮らないこと。"
            "・会話の中心は、育児をしている父親の「気持ち」と「行動」にすること。"
            "---"
            "【1ターン目の例】"
            "ユーザー：「最近育児と仕事の両立がうまくいかなくて…」"
            "あなた：「そっかワン、それは本当にがんばってるんだワンね。"
            "・どんなときにそう感じることが多いワン？"
            "・最近はどんな風に過ごしてるワン？」"

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

    # ✅ 月間カウントを加算（1往復）
    increment_user_count(user_id)

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
