
# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from importlib import reload
import json
import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = '40a7d339b7e86010'
APP_SECRET = 'SLi3bwgVZbaJaUXTCyZUcXClKWHrqa9n'


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def connect(q):
    data = {}
    data['to'] = 'zh-CHS'
    data['from'] = 'en'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    time.sleep(1)
    try:
        response = do_request(data)
        contentType = response.headers['Content-Type']
        res = json.loads(response.content)
        print(res['translation'][0])
        return res['translation'][0]
    except:
        return None


if __name__ == '__main__':
    save_data = []
    with open('classifier_dataset/Domain_Chinese/history_geograph_data.json') as f:
        data = json.load(f)
        for el in data:
            question_res = connect(el['question'])
            answer_res = connect(el['answer'])
            el['en_question'] = question_res
            el['en_answer'] = answer_res
            save_data.append(el)
            with open('classifier_dataset/Domain_Chinese/history_geograph_data_test.json', 'w') as f2:
                json.dump(save_data, f2, indent=4, ensure_ascii=False)


