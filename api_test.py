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
