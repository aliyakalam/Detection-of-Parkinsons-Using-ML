import pandas as pd
  

dataframe1 = pd.read_csv("data.txt")
  
# storing this dataframe in a csv file
dataframe1.to_csv('data.csv', 
                  index = None)