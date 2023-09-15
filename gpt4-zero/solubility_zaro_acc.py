import pandas as pd
pd.set_option('display.max_rows', 6000)
from PIL import Image
df=pd.read_excel(r"C:\Users\何思瑞\Desktop\gpt4-zero\solubility.xlsx",header=None,names=["result"])

df.dropna(inplace=True)
# df["result"]=df['result'].str.split(',', expand=True)[1]

# df["result"].dropna(inplace=True)

df = df[(df['result'].str.contains("target")) | (df['result'].str.contains("label"))]
# # print(df.reset_index())
df=df.reset_index(drop=True)
df["result"]=df['result'].str.split(',', expand=True)[1]

df["result"]=df['result'].str.split(':', expand=True)[1]
df["result"]=df["result"].str.replace(" ","")
# #
# # df.reset_index()
print(df)
step = 2
all=0
cor=0
unknow=0
for index, row in df.iterrows():
    if index % step == 0:
        # if index < 3996 and df.loc[index,"result"] != df.loc[index+1,"result"] :continue
        # else:print(index)
        # break
        if df.loc[index+1,"result"] == "-1":
            unknow+=1
            continue
        elif df.loc[index,"result"]==df.loc[index+1,"result"]:
            all+=1
            cor+=1
        else:all+=1
print(all)
print(unknow)
rate=cor/all
print(rate)
