# 评分模型的开发、部署、测试、文档说明全流程
**详见[github](https://github.com/chengsong990020186/xgboost_score_model/tree/master)**

https://github.com/chengsong990020186/xgboost_score_model/tree/master

**如对你有学习有帮助，请支持点赞~**

主要内容：
- 1.使用xgboost训练模型，并保存。
- 2.基于falsk框架，生成实时api接口,进行部署。
- 3.测试api接口。
- 4.api文档

## 1.使用xgboost训练模型，并保存。
训练数据已上传至github，可以自行下载。
```
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import logging

data=pd.read_csv('C:\\Users\\HP\\Desktop\\give me some credit\\data\\cs-training.csv')
del data['Unnamed: 0']
data.columns=['y','RevolvingUtilizationOfUnsecuredLines', 'age','NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome','NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
data_x=data[['RevolvingUtilizationOfUnsecuredLines', 'age','NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome','NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']]
data_y=data['y']
train_x, test_x, train_y, test_y = train_test_split(data_x.values, data_y.values, test_size=0.2,random_state=1234)
d_train = xgb.DMatrix(train_x, label=train_y)
d_valid = xgb.DMatrix(test_x, label=test_y)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#参数设置
params={
    'eta': 0.2, # 特征权重 取值范围0~1 通常最后设置eta为0.01~0.2
    'max_depth':3,   # 通常取值：3-10 树的深度
    'min_child_weight':1, # 最小样本的权重，调大参数可以防止过拟合
    'gamma':0.3,
    'subsample':0.8, #随机取样比例
    'colsample_bytree':0.8, #默认为1 ，取值0~1 对特征随机采集比例
    'booster':'gbtree', #迭代树
    'objective': 'binary:logistic', #逻辑回归，输出为概率
    # 'nthread':12, #设置最大的进程量，若不设置则会使用全部资源
    'scale_pos_weight': 1, #默认为0，1可以处理类别不平衡
    'lambda':1,   #默认为1
    'seed':1234, #随机数种子
    'silent':1 , #0表示输出结果
    'eval_metric': 'auc' # 检验指标
}
bst = xgb.train(params, d_train,1000,watchlist,early_stopping_rounds=500, verbose_eval=10)
tree_nums=bst.best_ntree_limit
logging.info('最优模型树的数量：%s，auc：%s' % (bst.best_ntree_limit, bst.best_score))
bst = xgb.train(params, d_train,tree_nums,watchlist,early_stopping_rounds=500, verbose_eval=10)
joblib.dump(bst, 'd:/xgboost.model') #保存模型
```

## 2.基于falsk框架，生成实时api接口,进行部署。
在目录建立 api.py 文件，代码如下
```
# -*- coding: utf-8 -*-
from flask import Flask, jsonify,request
import xgboost as xgb
from sklearn.externals import joblib
import numpy as np

import logging
import logging.handlers

logger = logging.getLogger()
file_handler = logging.handlers.TimedRotatingFileHandler('api_20190530.log', when='midnight')

# 测试按秒保留3个旧log文件
# file_handler = logging.handlers.TimedRotatingFileHandler("test.log", when='S', interval=1, backupCount=3)
# 设置后缀名称，跟strftime的格式一样
# file_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

#模型使用的特征

col_x=['RevolvingUtilizationOfUnsecuredLines', 
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans', 
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents']

xgboost_score_model=joblib.load('d:/xgboost.model')

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.config['JSON_AS_ASCII'] = False
@app.route('/xgboost_score_model/v1.0', methods=['POST'])
def create_task():
    logger.info(str(request.get_data()))
    try:
        request_data=request.get_json() #获取传入数据

        user_data=[request_data['user_data'][col] for col in col_x]
        result={}
        proba=xgboost_score_model.predict(xgb.DMatrix(np.array(user_data).reshape(1,-1)))[0] #模型预测概率

        B=20/(np.log(0.5))
        A=600-B*(np.log(1/9))
        score=A+B*np.log(proba/(1-proba)) #概率转化为评分

        result['code']=1
        result['user_id']=request_data['user_id']
        result['proba']=str(proba)
        result['score']=str(score)

        logger.info(result)
        return jsonify(result) #返回结果

    except KeyError as e:
        logger.info(e)
        return jsonify({'code':101,'error_type':'KeyError','error_msg':'输入特征错误:'+str(e)})

    except ValueError as e:
        logger.info(e)
        return jsonify({'code':102,'ValueError':'ValueError','error_msg':'输入值错误：'+str(e)})

    except Exception as e:
        logger.info(e)
        return jsonify({'code':103,'error_type':'Unchek_Exception','error_msg':'未知错误:'+str(e)})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)
```

在终端运行代码,进入Python文件目录，运行： 
```
python ./api.py
```
api接口开启成功，可以开始实时调用，并保存调用日志。

##  3.api接口测试。
### 3.1.postman进行测试

![postman进行测试](https://img-blog.csdnimg.cn/20190531165617975.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NzcWF6d3N4ZWRj,size_16,color_FFFFFF,t_70)

### 3.2.使用测试脚本
运行以下代码进行测试
```
import requests
import json
data_sample='''
{
    "user_id": "123456789",
    "user_data": {"DebtRatio": 0.80,
				 "MonthlyIncome": 9120.0,
				 "NumberOfDependents": 2.0,
				 "NumberOfOpenCreditLinesAndLoans": 13.0,
				 "NumberOfTime30-59DaysPastDueNotWorse": 2.0,
				 "NumberOfTime60-89DaysPastDueNotWorse": 0.0,
				 "NumberOfTimes90DaysLate": 0.0,
				 "NumberRealEstateLoansOrLines": 6.0,
				 "RevolvingUtilizationOfUnsecuredLines": 0.766,
				 "age": 45.0}
}
'''
data=json.loads(data_sample)
url='http://127.0.0.1:8000/xgboost_score_model/v1.0'
page=requests.post(url,json=data) 
print(page.json())

```

## 4.api文档

### xgboost评分模型调用接口
#### 接口描述
该接口是评分模型调用。

#### 接口说明

##### 1.请求URL:
```
测试环境：http://127.0.0.1:8000/xgboost_score_model/v1.0
正式环境：***
```
##### 2.请求方式
POST

##### 3.支持格式
JSON

##### 4.请求参数说明

参数名称|参数类型|描述
--|--|--
user_id|string|用户id
user_data|json|用户特征数据

##### 5.请求示例
```
{
    "user_id": "123456789",
    "user_data": {"DebtRatio": 0.8029821290000001,
				 "MonthlyIncome": 9120.0,
				 "NumberOfDependents": 2.0,
				 "NumberOfOpenCreditLinesAndLoans": 13.0,
				 "NumberOfTime30-59DaysPastDueNotWorse": 2.0,
				 "NumberOfTime60-89DaysPastDueNotWorse": 0.0,
				 "NumberOfTimes90DaysLate": 0.0,
				 "NumberRealEstateLoansOrLines": 6.0,
				 "RevolvingUtilizationOfUnsecuredLines": 0.76612660900000007,
				 "age": 45.0}
}
```

##### 6.成功返回的json对象说明
格式 : JSON 编码 : UTF-8 返回数据示例:
```
{
    "code": 1,
    "proba": "0.397148",
    "score": "548.644005001",
    "user_id": "123456789"
}
```

##### 7.失败返回时对应代码说明
```
{
    "code": 101,
    "error_msg": "输入特征错误:'aaaa'",
    "error_type": "KeyError"
}

{
    "ValueError": "ValueError",
    "code": 102,
    "error_msg": "输入值错误：invalid literal for int() with base 10: ''"
}
{
    "code": 103,
    "error_msg": "未知错误:400 Bad Request: Failed to decode JSON object: Expecting ',' delimiter: line 12 column 1 (char 227)",
    "error_type": "Unchek_Exception"
}
```
##### 8.返回字段说明
字段名|字段名称|描述
--|--|--
code|状态码|1：返回成功
user_id|用户id|用户id
proba|预测概率|预测概率
score|预测分数|预测分数


------------------------------------------------------------------------------------------------------------------------

**评分模型的开发、部署、测试、文档说明就基本完成了。如有疑问可与我联系：990020186@qq.com**
