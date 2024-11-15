import streamlit as st
from streamlit_chat import message
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 時間管理
from time import sleep
import datetime
import pytz # タイムゾーン
global now # PCから現在時刻
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# firebase
import firebase_admin
from google.oauth2 import service_account
from google.cloud import firestore
import json
import tiktoken

# プロンプト
prompt_list = ["preprompt_affirmative_individualizing_nuclear.txt", "preprompt_negative_binding_nuclear.txt"]
# モデル
model_list = ["gpt-4-1106-preview", "gpt-4o", "gpt-4o-mini"]
# 待機時間
sleep_time_list = [5, 5, 5, 5, 5, 5, 5, 5]
# 表示テキスト
text_list = ['「原子力発電を廃止すべきか否か」という意見に対して、あなたの意見を入力し、送信ボタンを押してください。', 'あなたの意見を入力し、送信ボタンを押してください。']

# メモリ設定
if not "memory" in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=8, return_messages=True)

# ID入力※テスト用フォーム
def input_id():
    if not "user_id" in st.session_state:
        st.session_state.user_id = "hogehoge"
    with st.form("id_form", enter_to_submit=False):
        # prompt_option = st.selectbox(
            # "プロンプトファイル選択※テスト用フォーム",
            # ("{}".format(prompt_list[0]), "{}".format(prompt_list[1])),)
        model_option = st.selectbox(
            "モデル選択※テスト用フォーム",
            ("{}".format(model_list[0]), "{}".format(model_list[1]), "{}".format(model_list[2])),)
        user_id = st.text_input('学籍番号を入力し、送信ボタンを押してください')
        submit_id = st.form_submit_button(
            label="送信",
            type="primary")
    if submit_id:
        with open(prompt_list[1], 'r', encoding='utf-8') as f:
            st.session_state.systemprompt = f.read()
        st.session_state.model = model_option
        st.session_state.user_id = str(user_id)
        st.session_state.state = 2
        st.rerun()

# プロンプト設定
if "systemprompt" in st.session_state:
    template = st.session_state.systemprompt # st.session_state.systemprompt
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

# モデル設定
if "model" in st.session_state:
    # モデルのインスタンス生成
    chat = ChatOpenAI(
        model=st.session_state.model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=0,
        api_key= st.secrets.openai_api_key
    )
    # チェインを設定
    conversation = ConversationChain(llm=chat, memory=st.session_state.memory, prompt=st.session_state.prompt)
    encoding = tiktoken.encoding_for_model(st.session_state.model)


# Firebase 設定の読み込み
key_dict = json.loads(st.secrets["firebase"]["textkey"])
creds = service_account.Credentials.from_service_account_info(key_dict)
project_id = key_dict["project_id"]
db = firestore.Client(credentials=creds, project=project_id)

# 入力時の動作
def click_to_submit():
    # 待機中にも履歴を表示
    chat_placeholder = st.empty()
    with chat_placeholder.container(height=600):
        for msg in st.session_state.log:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, avatar_style="adventurer", seed="Nala")
            else:
                message(msg["content"], is_user=False, avatar_style="micah")
    with st.spinner("相手からの返信を待っています..."):
        st.session_state.send_time = str(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        st.session_state.response = conversation.predict(input=st.session_state.user_input)
        # count token
        # if not "total_output_tokens" in st.session_state:
            # st.session_state.total_output_tokens = 0
        # st.session_state.output_tokens = len(encoding.encode(st.session_state.response))
        # st.session_state.total_output_tokens += st.session_state.output_tokens
        st.session_state.log.append({"role": "AI", "content": st.session_state.response})
        sleep(sleep_time_list[st.session_state.talktime])
        st.session_state.return_time = str(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        doc_ref = db.collection(str(st.session_state.user_id)).document(str(st.session_state.talktime))
        doc_ref.set({
            "Human": st.session_state.user_input,
            "AI": st.session_state.response,
            "Human_msg_sended": st.session_state.send_time,
            "AI_msg_returned": st.session_state.return_time,
        })
        st.session_state.talktime += 1
        st.session_state.state = 2
        st.rerun()

# チャット画面
def chat_page():
    if not "talktime" in st.session_state:
        st.session_state.talktime = 0
    if not "log" in st.session_state:
        st.session_state.log = []
    # 履歴表示
    chat_placeholder = st.empty(height=600)
    with chat_placeholder.container():
        for msg in st.session_state.log:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, avatar_style="adventurer", seed="Nala")
            else:
                message(msg["content"], is_user=False, avatar_style="micah")
        # print token
        # if "input_tokens" in st.session_state:
            # st.write("input tokens : {}※テスト用".format(st.session_state.input_tokens))
            # st.write("output tokens : {}※テスト用".format(st.session_state.output_tokens))
    # 入力フォーム
    if st.session_state.talktime < 5:
        if not "user_input" in st.session_state:
            st.session_state.user_input = "hogehoge"
        with st.container():
            with st.form("chat_form", clear_on_submit=True, enter_to_submit=False):
                if st.session_state.talktime == 0:
                    user_input = st.text_area(text_list[0])
                else:
                    user_input = st.text_area(text_list[1])
                submit_msg = st.form_submit_button(
                    label="送信",
                    type="primary")
            if submit_msg:
                st.session_state.user_input = user_input
                st.session_state.log.append({"role": "user", "content": st.session_state.user_input})
                # count token
                # if not "total_input_tokens" in st.session_state:
                    # st.session_state.total_input_tokens = 0
                # st.session_state.input_tokens = 0
                # system_tokens = encoding.encode(template)
                # st.session_state.input_tokens += len(system_tokens)
                # st.session_state.total_input_tokens += len(system_tokens)
                # for msg in st.session_state.log:
                    # tokens = encoding.encode(msg["content"])
                    # st.session_state.input_tokens += len(tokens)
                    # st.session_state.total_input_tokens += len(tokens)
                st.session_state.state = 3
                st.rerun()
    elif st.session_state.talktime == 5: # 終了時の動作
        url = "https://nagoyapsychology.qualtrics.com/jfe/form/SV_87jQ6Hj2rjLDdSm"
        # print total token counts
        # st.write("total input tokens : {}※テスト用".format(st.session_state.total_input_tokens))
        # st.write("total output tokens : {}※テスト用".format(st.session_state.total_output_tokens))
        st.markdown(
            f"""
            会話が規定回数に達しました。\n\n
            以下の"アンケートに戻る"をクリックして、アンケートに回答してください。\n\n
            アンケートページは別のタブで開きます。\n\n
            <a href="{url}" target="_blank">アンケートに戻る</a>
            """,
            unsafe_allow_html=True)

def main():
    hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    if not "state" in st.session_state:
        st.session_state.state = 1
    if st.session_state.state == 1:
        input_id()
    elif st.session_state.state == 2:
        chat_page()
    elif st.session_state.state == 3:
        click_to_submit()

if __name__ == "__main__":
    main()

"""

# streamlit 
import streamlit as st
from streamlit_chat import message

from operator import itemgetter
from typing import List

# langchain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from typing import Optional

if "history" not in st.session_state:
    st.session_state.history_instance = InMemoryHistory(name="Streamlit User")
if "store" not in st.session_state:
    st.session_state.store = {}

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

# セッションIDごとの会話履歴の取得
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# RunnableWithMessageHistoryの準備
st.session.state_runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# RunnableWithMessageHistoryの準備
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
print(store)  # noqa: T201

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=210,
    timeout=None,
    max_retries=0,
    api_key= st.secrets.openai_api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", {systemprompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}"),
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="user_input",
    history_messages_key="history",
)

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# 時間管理
from time import sleep
import datetime
import pytz # タイムゾーン
global now # PCから現在時刻
now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

# firebase
import firebase_admin
from google.oauth2 import service_account
from google.cloud import firestore
import json

# クエリ取得
params = st.query_params()

#
# 例： http:/hogehoge?param1=euthanasia&param2=2
# params = {
#   "param1": [
#     "euthanasia"
#   ],
#   "params": [
#     "2"
#   ]
# }

# プロンプト
prompt_list = ["prompt.txt"]
# 待機時間
# sleep_time_list = [60, 75, 75, 90, 60]
sleep_time_list = [5, 5, 5, 5, 5]

# モデルのインスタンス生成
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=210,
    timeout=None,
    max_retries=0,
    api_key= st.secrets.openai_api_key
)

if not "memory" in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=8, return_messages=True)

# ID入力
def input_id():
    if not "user_id" in st.session_state:
        st.session_state.user_id = "hogehoge"
    with st.form("id_form", enter_to_submit=False):
        option = st.selectbox(
            "プロンプトファイル選択※テスト用フォーム",
            ("{}".format(prompt_list[0])))
        user_id = st.text_input('idを入力してください')
        submit_id = st.form_submit_button(
            label="送信",
            type="primary")
    if submit_id:
        st.session_state.user_id = str(user_id)
        fname = option
        with open(fname, 'r', encoding='utf-8') as f:
            st.session_state.systemprompt = f.read()
        st.session_state.state = 2
        st.rerun()

# プロンプト設定
if "systemprompt" in st.session_state:
    template = st.session_state.systemprompt
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    # チェインを設定
    conversation = ConversationChain(llm=chat, memory=st.session_state.memory, prompt=st.session_state.prompt)

# Firebase 設定の読み込み
key_dict = json.loads(st.secrets["firebase"]["textkey"])
creds = service_account.Credentials.from_service_account_info(key_dict)
project_id = key_dict["project_id"]
db = firestore.Client(credentials=creds, project=project_id)


# 入力時の動作
def click_to_submit():
    # 待機中にも履歴を表示
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for msg in st.session_state.log:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, avatar_style="adventurer", seed="Nala")
            else:
                message(msg["content"], is_user=False, avatar_style="micah")
    with st.spinner("相手の返信を待っています…。"):
        st.session_state.send_time = str(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        response = runnable_with_history.invoke(
            {"input": st.session_state.user_input},
            config={"configurable": {"session_id": st.session_state.user_id}},
        )
        history.add_message(AIMessage(content="hello"))
        st.session_state.response = conversation.predict(input=st.session_state.user_input)
        # st.session_state.memory.save_context({"input": st.session_state.user_input}, {"output": st.session_state.response})
        st.session_state.log.append({"role": "AI", "content": st.session_state.response})
        sleep(sleep_timelist[st.session_state.talktime])
        st.session_state.return_time = str(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        doc_ref = db.collection(str(st.session_state.user_id)).document(str(st.session_state.talktime))
        doc_ref.set({
            "Human": st.session_state.user_input,
            "AI_": st.session_state.response,
            "Human_meg_sended": st.session_state.send_time,
            "AI_meg_returned": st.session_state.return_time,
        })
        st.session_state.talktime += 1
        st.session_state.state = 2
        st.rerun()

# チャット画面
def chat_page():
    if not "talktime" in st.session_state:
        st.session_state.talktime = 0
    if not "log" in st.session_state:
        st.session_state.log = []
    st.markdown("日本は原子力発電を廃止すべきかどうかについて、あなたの意見を述べ、議論を行ってください")
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for msg in st.session_state.log:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, avatar_style="adventurer", seed="Nala")
            else:
                message(msg["content"], is_user=False, avatar_style="micah")
    if st.session_state.talktime < 5:
        if not "user_input" in st.session_state:
            st.session_state.user_input = "hogehoge"
        with st.container():
            with st.form("chat_form", clear_on_submit=True, enter_to_submit=False):
                user_input = st.text_area('意見を入力して下さい')
                submit_msg = st.form_submit_button(
                    label="送信",
                    type="primary")
            if submit_msg:
                st.session_state.user_input = user_input
                st.session_state.log.append({"role": "user", "content": st.session_state.user_input})
                st.session_state.state = 3
                st.rerun()
    elif st.session_state.talktime == 5:
        url = "https://www.nagoya-u.ac.jp/"
        # url = "https://survey.qualtrics.com/jfe/form/SV_123456789?ソース=Facebook&Campaign=モバイル"
        st.markdown(
            f"""
            会話は終了しました。以下のリンクをクリックしてアンケートに回答してください。  
            <a href="{url}" target="_blank">こちら</a>
            """,
            unsafe_allow_html=True)

def main():
    hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    if not "state" in st.session_state:
        st.session_state.state = 1
    if st.session_state.state == 1:
        input_id()
    elif st.session_state.state == 2:
        chat_page()
    elif st.session_state.state == 3:
        click_to_submit()

if __name__ == "__main__":
    st.title('チャット対話実験：先攻')
    main()
"""
