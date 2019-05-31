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
