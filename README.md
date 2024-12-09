# AIoT-DA_HW4-1_-Pycarat-to-compare-ML-agorithms-on-classification-problem-16-Model
Pycarat to compare ML agorithms on classification problem 16 Model (multi-features titanic is ok , ) 
-------------------------------------------
#the submitted score of my prediction on kaggle is 0.73684
#注意train.csv和test.csv放的路徑要記得修改

#prompt1
我現在要用colab做一個關於Pycarat to compare ML agorithms on classification problem 16 Model並用Titanic訓練集，幫我回答我等等的問題

#prompt2
我如果要做到Pycarat to compare ML agorithms on classification problem 16 Model (multi-features titanic)這項題目的話 我需要下載哪些工具

#prompt3
這是我目前的code，並遇到以下問題
Traceback (most recent call last):
  File "d:\vs code\AIoT_Project\AutoML_Ensemble model optimization\4-1.py", line 103, in <module>
    stacker = stack_models(estimator_list = [ridge,lda,gbc], meta_model = lr)
NameError: name 'lr' is not defined

#prompt4
能幫我將模型儲存成.csv檔嗎

#prompt5
繳交到kaggle時出現這些格式問題

#prompt6
Traceback (most recent call last):
  File "d:\vs code\AIoT_Project\AutoML_Ensemble model optimization\4-1-1.py", line 92, in <module>
    submission = predictions[['PassengerId', 'Label']]
  File "C:\Users\user\anaconda3\envs\aiothwenv\lib\site-packages\pandas\core\frame.py", line 3899, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "C:\Users\user\anaconda3\envs\aiothwenv\lib\site-packages\pandas\core\indexes\base.py", line 6115, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "C:\Users\user\anaconda3\envs\aiothwenv\lib\site-packages\pandas\core\indexes\base.py", line 6179, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['Label'] not in index"

#prompt7
幫我加上18個演算法的 stacking 模型，並使用 xgboost 作第二層預測

-->result
