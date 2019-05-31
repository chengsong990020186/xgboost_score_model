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
