import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pycaret.classification import *

# 设置 Matplotlib 的显示风格
plt.style.use('seaborn')

# 加载数据集
try:
    data_train = pd.read_csv(r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\train.csv")
    data_test = pd.read_csv(r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\test.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 特征处理函数
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

# 转换特征
data_train = transform_features(data_train)
data_test = transform_features(data_test)

# 数据拆分
train_data, test_data = train_test_split(data_train.drop(['PassengerId'], axis=1), 
                                         random_state=100, 
                                         train_size=0.8)

# PyCaret 设置
clf1 = setup(data=train_data, 
             target='Survived', 
             categorical_features=['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Lname', 'NamePrefix'])

# 创建基础模型，使用 5 折交叉验证
lr = create_model('lr', fold=5)
knn = create_model('knn', fold=5)
nb = create_model('nb', fold=5)
dt = create_model('dt', fold=5)
svm = create_model('svm', fold=5)
rbfsvm = create_model('rbfsvm', fold=5)
gpc = create_model('gpc', fold=5)
mlp = create_model('mlp', fold=5)
ridge = create_model('ridge', fold=5)
rf = create_model('rf', fold=5)
qda = create_model('qda', fold=5)
ada = create_model('ada', fold=5)
lda = create_model('lda', fold=5)
gbc = create_model('gbc', fold=5)
et = create_model('et', fold=5)
xgboost = create_model('xgboost', fold=5)
lightgbm = create_model('lightgbm', fold=5)
catboost = create_model('catboost', fold=5)

# 堆叠模型，使用 xgboost 作为元模型
stacker_all = stack_models(estimator_list=[
    lr, knn, nb, dt, svm, rbfsvm, gpc, mlp, ridge, rf, qda, ada, lda, gbc, et, xgboost, lightgbm, catboost
], meta_model=xgboost)

# 使用堆叠模型预测
predictions = predict_model(stacker_all, data=data_test)

# 调试：打印预测结果的列名
print("预测结果的列名：", predictions.columns)

# 提取 PassengerId 和预测结果列
if 'Label' in predictions.columns:
    submission = predictions[['PassengerId', 'Label']]
    submission.rename(columns={'Label': 'Survived'}, inplace=True)
elif 'prediction_label' in predictions.columns:
    submission = predictions[['PassengerId', 'prediction_label']]
    submission.rename(columns={'prediction_label': 'Survived'}, inplace=True)
else:
    raise ValueError("无法找到预测列，请检查预测结果的列名！")

# 保存预测结果为 CSV 文件
output_path = r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\predictions.csv"
submission.to_csv(output_path, index=False)
print(f"预测结果已保存到：{output_path}")

# 计算准确率（如果测试集包含真实标签）
if 'Survived' in data_test.columns:
    accuracy = accuracy_score(data_test['Survived'], predictions['Label'])
    print(f"预测准确率: {accuracy}")
else:
    print("测试数据集中没有真实标签，无法计算准确率。")
