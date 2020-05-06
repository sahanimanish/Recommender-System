# import pickle


# class Test:
#     name = "Rajiv Prajapati"
#     college = "Gu"
#     age = 23
#     def __init__(self):
#         self.a = 20
#         self.seta()
#     def seta(self):
#         self.a = 420
#     def display(self):
#         print(f"name = {self.name} college={self.college} age={self.age} a = {self.a}")

# # t = Test()
# # fl = open('testdb.pkl','wb')
# # pickle.dump(t,fl)
# # fl.close()


# fl = open('testdb.pkl', 'rb')
# testObj = pickle.load(fl)
# testObj.display()
# print(type(testObj))
# fl.close()


# import pandas as pd
# import numpy as np

# df = pd.read_csv('./data/ml-latest-small/ratings.csv')
# print(df.head(5))

import pickle
