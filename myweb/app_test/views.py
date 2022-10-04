from urllib import request
from django import http
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
import json
# Create your views here.
from datetime import datetime
import os
from pandas import *
import pandas as pd
from numpy.ma.core import count
from pandas.core.frame import DataFrame
data = pd.read_csv (r'C:\Users\User\Desktop\GP\data.csv')
dataset = data.drop(['id','business_name','street_name','owner','business_type'], axis = 1,inplace=True)
#cross between 2 colomns to find probability of the cost in each business
df1=pd.DataFrame(data, columns=['business_number','cost'])
d1 = pd.crosstab(index=df1["business_number"], columns="count") 
d2=pd.crosstab(index=df1["cost"], columns="count") 
d3=pd.crosstab(index=df1["cost"], columns=df1["business_number"],margins=True ) 
d4=d3/d3.loc["All","All"]
df2 = pd.DataFrame(data, columns=['business_number'])
df2.drop_duplicates()
business_array = df2.drop_duplicates()[['business_number']].to_numpy()
for t in range(92):
  if t==0:
    df4 = pd.DataFrame(columns=business_array[t], index=range(3))
  else:
    df4.insert(t, business_array[t],0, True)
df4.rename(index = {0: "main_street", 1:"main-substreet",2:"substreet"},inplace = True)
df3 = pd.DataFrame(data, columns=['street_number'])
street =df3[['street_number']].to_numpy()
job = df2[['business_number']].to_numpy()
rows, cols = (2, 1056)
matrix= [[0 for i in range(cols)] for j in range(rows)]
count1 =0
count2 =0
count3 =0
sum1 =0
sum2 =0
sum3 =0
st1 =[0 for x in range(92)]
st2 =[0 for x in range(92)]
st3 =[0 for x in range(92)]
for j in range(1056):
  l = street[j]%10
  matrix[0][j] = l
  matrix[1][j] = job[j]
for c in range(92):
  k=business_array[c]
  count1=0
  count2=0
  count3=0
  for p in range(1056):
    if k == matrix[1][p] :
      if matrix[0][p] == 1:
        count1 = count1 +1
      if matrix[0][p] == 2:
        count2 = count2 +1
      elif matrix [0][p] == 3:
        count3 = count3 + 1
  st1[c]=count1
  st2[c]=count2
  st3[c]=count3
  sum1=sum1 + count1
  sum2=sum2 + count2
  sum3=sum3 +count3
df4.loc["main_street"] = st1
df4.loc["main-substreet"] = st2
df4.loc["substreet"] =st3
df5=df4/df4.sum()
#cross between 2 colomns to business number and the area to find the similarity
darea=pd.DataFrame(data, columns=['business_number','area'])
dlast=pd.crosstab(index=darea["area"], columns=darea["business_number"],margins=True ) 
dlast = dlast.drop("All", axis=0)
sorted = dlast.sort_values(by=['All'])
dup = dlast.pivot_table(columns=['All'], aggfunc='size')
# dataframe of sorted buisness
sortedBuisness = sorted.loc[:,'All']
# dataframe of sorted areas
sortedAreas = sorted.iloc[:,:1]
Array3 = sortedAreas.index
# areas classification 
# #the cost with area
dk1=pd.DataFrame(data, columns=['area','cost'])
dk1
#frequency table for cost with area
dk=pd.crosstab(index=dk1["cost"], columns=dk1["area"],margins=True ) 
dk
#find the cost sum for every area 
dk2 = dk1.groupby(['area']).agg({'cost': 'sum'})
#add the cost number to th data frame 
dk2.insert(1, "cost_count",dk.loc['All'], True)
#the mean cost for every area
dk3 = dk2['cost']/dk2['cost_count']
dk2.insert(2, "cost_avg",dk3, True)
dk4=dk2.iloc[:,:0]
#collect all i want to use in one datafram
dk5=pd.DataFrame(dk4.index, columns=['area'])
dk2.insert(3, "area",dk5, True)
dk2.reset_index(inplace = True, drop = True)
dk2.rename(columns = {'cost_count':'job_count'}, inplace = True)
#k-mean (area , cost avg ,job num)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
X = dk2.iloc[:,1:4]
k_means_optimum = KMeans(n_clusters = 4, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)
dk2['cluster'] = y  
data1 = dk2[dk2.cluster==0]
data2 = dk2[dk2.cluster==1]
data3 = dk2[dk2.cluster==2]
data4 = dk2[dk2.cluster==3]
list1=[]
Tlist1=dk2.loc[dk2['cluster'] == 0]
list1=Tlist1['area'].values
list1
list2=[]
Tlist2=dk2.loc[dk2['cluster'] == 1]
list2=Tlist2['area'].values
list2
list3=[]
Tlist3=dk2.loc[dk2['cluster'] == 2]
list3=Tlist3['area'].values
list3
list4=[]
Tlist4=dk2.loc[dk2['cluster'] == 3]
list4=Tlist4['area'].values
list4
acc=sorted.drop("All", axis=1)
#take the areas from list1 and show business in each area
x1 = len(list1)
rows, cols = (x1, 92)
matrix1= [[0 for i in range(cols)] for j in range(rows)]
for i  in range(x1):
  k=list1[i]
  matrix1[i] = acc.loc[k]
m1=pd.DataFrame(matrix1)
#list2 --> m2
x2 = len(list2)
rows, cols = (x2, 92)
matrix2= [[0 for i in range(cols)] for j in range(rows)]
for i  in range(x2):
  k=list2[i]
  matrix2[i] = acc.loc[k]
m2=pd.DataFrame(matrix2)
#list3 -->m3
x3 = len(list3)
rows, cols = (x3, 92)
matrix3= [[0 for i in range(cols)] for j in range(rows)]
for i  in range(x3):
  k=list3[i]
  matrix3[i] = acc.loc[k]
m3=pd.DataFrame(matrix3)
#list4 -->m4
x4 = len(list4)
rows, cols = (x4, 92)
matrix4= [[0 for i in range(cols)] for j in range(rows)]
for i  in range(x4):
  k=list4[i]
  matrix4[i] = acc.loc[k]
m4=pd.DataFrame(matrix4)
#find the max with max_id(the area)  -->m1max
maxValueIndex = m1.idxmax()
m1max=pd.DataFrame(maxValueIndex)
m1max['max'] = m1.max()
m1max.rename(columns = {0:'max_id'},inplace = True)
#remove max = 0 ... -->m1max1
a1=[]
for i in  range(92):
  val =m1max['max'] .iloc[i]
  if val == 0:
    a1.append(i)
m1max1 = m1max.drop(m1max.index[a1])
#find the max with max_id(the area) -->m2max
maxValueIndex2 = m2.idxmax()
m2max=pd.DataFrame(maxValueIndex2)
m2max['max'] = m2.max()
m2max.rename(columns = {0:'max_id'},inplace = True)
#remove max = 0 ... -->m2max2
a2=[]
for i in  range(92):
  val =m2max['max'] .iloc[i]
  if val == 0:
    a2.append(i)
m2max2 = m2max.drop(m2max.index[a2])
#find the max with max_id(the area)  -->m3max
maxValueIndex3 = m3.idxmax()
m3max=pd.DataFrame(maxValueIndex3)
m3max['max'] = m3.max()
m3max.rename(columns = {0:'max_id'},inplace = True)
#remove max = 0 ... -->m3max3
a3=[]
for i in  range(92):
  val =m3max['max'] .iloc[i]
  if val == 0:
    a3.append(i)
m3max3 = m3max.drop(m3max.index[a3])
#find the max with max_id(the area) -->m4max
maxValueIndex4 = m4.idxmax()
m4max=pd.DataFrame(maxValueIndex4)
m4max['max'] = m4.max()
m4max.rename(columns = {0:'max_id'},inplace = True)
#remove max = 0 ... -->m4max4
a4=[]
for i in  range(92):
  val =m4max['max'].iloc[i]
  if val == 0:
    a4.append(i)
m4max4 = m4max.drop(m4max.index[a4])
#first step predicting how much i need business in each area
#area_need =(max_number-n)/max_number
buss = df2.drop_duplicates()[['business_number']].to_numpy()
d1 = pd.DataFrame(data, columns=['area'])
area = d1.drop_duplicates()[['area']].to_numpy()
for r in range(92):
  if r==0:
    d8 = pd.DataFrame(columns=buss[r], index=list1)
  else:
    d8.insert(r, buss[r],0, True)
area_buss1=d8

for r in range(92):
  if r==0:
    d9 = pd.DataFrame(columns=buss[r], index=list2)
  else:
    d9.insert(r, buss[r],0, True)
area_buss2=d9

for r in range(92):
  if r==0:
    d10 = pd.DataFrame(columns=buss[r], index=list3)
  else:
    d10.insert(r, buss[r],0, True)
area_buss3=d10

for r in range(92):
  if r==0:
    d11 = pd.DataFrame(columns=buss[r], index=list4)
  else:
    d11.insert(r, buss[r],0, True)
area_buss4=d11
x=list(area_buss1.columns)
y=list(area_buss1.index)
r1=sorted.loc[y,x]
x=list(area_buss2.columns)
y=list(area_buss2.index)
r2=sorted.loc[y,x]
x=list(area_buss3.columns)
y=list(area_buss3.index)
r3=sorted.loc[y,x]
x=list(area_buss4.columns)
y=list(area_buss4.index)
r4=sorted.loc[y,x]
m1=m1max.drop(labels='max_id',axis=1)
m2=m2max.drop(labels='max_id',axis=1)
m3=m3max.drop(labels='max_id',axis=1)
m4=m4max.drop(labels='max_id',axis=1)
e=0
for i in r1.index:
  for j in r1.columns:
    n=r1.loc[i,j]
    max=m1.loc[j,'max']
    if  max!=0:
      p=(max-n)/max
      r1.loc[i,j]=p
    else:
      r1.loc[i,j]=0
e=0
for i in r2.index:
  for j in r2.columns:
    n=r2.loc[i,j]
    max=m2.loc[j,'max']
    if  max!=0:
      p=(max-n)/max
      r2.loc[i,j]=p
    else:
      r2.loc[i,j]=0
e=0
for i in r3.index:
  for j in r3.columns:
    n=r3.loc[i,j]
    max=m3.loc[j,'max']
    if  max!=0:
      p=(max-n)/max
      r3.loc[i,j]=p
    else:
      r3.loc[i,j]=0
e=0
for i in r4.index:
  for j in r4.columns:
    n=r4.loc[i,j]
    max=m4.loc[j,'max']
    if  max!=0:
      p=(max-n)/max 
      r4.loc[i,j]=p
    else:
      r4.loc[i,j]=0
#to find the areas have all the streets types
df98 = pd.DataFrame(data, columns=['area'])
area_street_array = df98.drop_duplicates()[['area']].to_numpy()
for t in range(71):
  if t==0:
    df99 = pd.DataFrame(columns=area_street_array[t], index=range(3))
  else:
    df99.insert(t, area_street_array[t],0, True)
df99.rename(index = {0: "main_street", 1:"main-substreet",2:"substreet"},inplace = True)
df3 = pd.DataFrame(data, columns=['street_number'])
street =df3[['street_number']].to_numpy()
a = df98[['area']].to_numpy()
rows, cols = (2, 1056)
matrix= [[0 for i in range(cols)] for j in range(rows)]
count1 =0
count2 =0
count3 =0
st11 =[0 for x in range(71)]
st22 =[0 for x in range(71)]
st33 =[0 for x in range(71)]
for j in range(1056):
  l = street[j]%10
  matrix[0][j] = l
  matrix[1][j] = a[j]
for c in range(71):
  k=area_street_array[c]
  count1=0
  count2=0
  count3=0
  for p in range(1056):
    if k == matrix[1][p] :
      if matrix[0][p] == 1:
        count1 = 1
      if matrix[0][p] == 2:
        count2 = 1
      elif matrix [0][p] == 3:
        count3 =  1
  st11[c]=count1
  st22[c]=count2
  st33[c]=count3
df99.loc["main_street"] = st11
df99.loc["main-substreet"] = st22
df99.loc["substreet"] =st33
from pandas.core.frame import DataFrame
def streets_classifier(area,street_cat,cost):
    df_mat1 = DataFrame ()
    df_mat = DataFrame ()
    k = 0
    area_val = []  
    street_val = []
    AV_list =[]
    mul_list = []
    # sort business values
    b1 = data[['business_number']].to_numpy()
    b2 = DataFrame(b1)
    b3 = b2.drop_duplicates()
    b4 = b3.sort_values(by=0)
    list_b2 = b4.values.tolist()
    #display(list_b2)
    #areas classification
    # list 1 
    for i in list1:
        if ( area == i ):
            for j in r1.index:
                k=k+1
                if ( area == j):
                    area_val = r1.iloc[k-1,0:]
                    AV_list = area_val.to_list()
    # list 2 
    for i in list2:
        if ( area == i ):
            for j in r2.index:
                k=k+1
                if ( area == j):
                    area_val = r2.iloc[k-1,0:]
                    AV_list = area_val.to_list()
    # list 3 
    for i in list3:
        if ( area == i ):
            for j in r3.index:
                k=k+1
                if ( area == j):
                    area_val = r3.iloc[k-1,0:]
                    AV_list = area_val.to_list()
    # list 4
    for i in list4:
        if ( area == i ):
            for j in r4.index:
                k=k+1
                if ( area == j):
                    area_val = r4.iloc[k-1,0:]
                    AV_list = area_val.to_list()
    # street classification :         
    # df5 -> dataframe of the streets category probabilities 
    k = 0 
    for q in df5.index:
        k=k+1
        if ( street_cat == q ):
            j = df5.iloc[k-1,0:]
            street_val = j.iloc[0:]
            SV_list = street_val.to_list()  
    # cost classification : 
    # the pobabilites are in dataframe d4 
    cost_val = []
    cost_val2 = [] 
    cost1 = 0
    # cost 
    k = 0 
    for i in d4.index:
        cost1 = i 
        if ( cost == i ):
            if (i == 5 ):
                cost_val5 = d4.iloc[i-1,0:]
                CV_list5 = cost_val5.to_list()
            elif (i == 4):
                cost_val4 = d4.iloc[i-1,0:]
                CV_list4 = cost_val4.to_list()
                cost_val5 = d4.iloc[i,0:]
                CV_list5 = cost_val5.to_list()
            elif (i == 3):
                cost_val3 = d4.iloc[i-1,0:]
                CV_list3 = cost_val3.to_list()
                cost_val4 = d4.iloc[i,0:]
                CV_list4 = cost_val4.to_list()
                cost_val5 = d4.iloc[i+1,0:]
                CV_list5 = cost_val5.to_list()
            elif (i == 2):
                cost_val2 = d4.iloc[i-1,0:]
                CV_list2 = cost_val2.to_list()
                cost_val3 = d4.iloc[i,0:]
                CV_list3 = cost_val3.to_list()
                cost_val4 = d4.iloc[i+1,0:]
                CV_list4 = cost_val4.to_list()
                cost_val5 = d4.iloc[i+2,0:]
                CV_list5 = cost_val5.to_list()
            elif (i == 1):
                cost_val = d4.iloc[i-1,0:]
                CV_list = cost_val.to_list()
                cost_val2 = d4.iloc[i,0:]
                CV_list2 = cost_val2.to_list()
                cost_val3 = d4.iloc[i+1,0:]
                CV_list3 = cost_val3.to_list()
                cost_val4 = d4.iloc[i+2,0:]
                CV_list4 = cost_val4.to_list()
                cost_val5 = d4.iloc[i+3,0:]
                CV_list5 = cost_val5.to_list()
            else :
                print(" bad input ")

    # we got the vectors we need 
    # sort the vectors in one matrix 
    mul_list = []
    mul_list3 = []
    counter1 = 0 
    for i in range(len(AV_list)):
        mul_list.append(AV_list[i]*SV_list[i])
    rslt = DataFrame(mul_list)
    mul_list1 = rslt #.sort_values(by=0)
    mul_list2 =  mul_list1.values.tolist()
    MAT1 = [[ 0 for i in range (5)]  for i in range (92)]
    # for i in mul_list2:
    #  MAT1[1].append(i) 
    h = 0 
    for item in mul_list2:
        for h in item:
            float(h)
            MAT1[0].append(h)
    i = 0 
    if ( cost == 1):
        for i in CV_list:
            MAT1[1].append(i)
        for i in CV_list2:
            MAT1[2].append(i)
        for i in CV_list3:
            MAT1[3].append(i)
        for i in CV_list4:
            MAT1[4].append(i)
        for i in CV_list5:
            MAT1[5].append(i)
        df_mat = df_mat.append(MAT1)
        df_mat1 = df_mat1.append(df_mat.iloc[:6,0:])
        df_mat1.rename(index={ 0 : 'streetArea_prob',1 : 'cost_prob_1',2 : 'cost_prob_2',3 : 'cost_prob_3',4 : 'cost_prob_4' ,5 : 'cost_prob_5'}, inplace = True)
    elif ( cost == 2 ):
        for i in CV_list2:
            MAT1[1].append(i)
        for i in CV_list3:
            MAT1[2].append(i)
        for i in CV_list4:
            MAT1[3].append(i)
        for i in CV_list5:
            MAT1[4].append(i)
        df_mat = df_mat.append(MAT1)
        df_mat1 = df_mat1.append(df_mat.iloc[:5,0:])
        df_mat1.rename(index={ 0 : 'streetArea_prob', 1 : 'cost_prob_2',2 : 'cost_prob_3',3 : 'cost_prob_4',4 : 'cost_prob_5'}, inplace = True)
    elif ( cost == 3 ):
        for i in CV_list3:
            MAT1[1].append(i)
        for i in CV_list4:
            MAT1[2].append(i)
        for i in CV_list5:
            MAT1[3].append(i)
        df_mat = df_mat.append(MAT1)
        df_mat1 = df_mat1.append(df_mat.iloc[:4,0:])
        df_mat1.rename(index={ 0 : 'streetArea_prob', 1 : 'cost_prob_3',2 : 'cost_prob_4',3 : 'cost_prob_5'}, inplace = True)
    elif ( cost == 4 ):
        for i in CV_list4:
            MAT1[1].append(i)
        for i in CV_list5:
            MAT1[2].append(i)
        df_mat = df_mat.append(MAT1)
        df_mat1 = df_mat1.append(df_mat.iloc[:3,0:])
        df_mat1.rename(index={ 0 : 'streetArea_prob', 1 : 'cost_prob_4',2 : 'cost_prob_5'}, inplace = True)
    elif ( cost == 5 ):
        for i in CV_list5:
            MAT1[1].append(i)
        df_mat = df_mat.append(MAT1)
        df_mat1 =df_mat1.append(df_mat.iloc[:2,0:])
        df_mat1.rename(index={ 0 : 'streetArea_prob', 1 : 'cost_prob_5'},inplace = True)
    C4 = 5
    for x in df5.columns:
        df_mat1.rename(columns={ C4 : x}, inplace = True)
        C4 += 1

    for i in  df_mat1.index:
        for j in df_mat1.columns:
            if df_mat1.loc['streetArea_prob',j] == 0 :
                df_mat1.drop(j, inplace=True, axis=1)
    df_mat1.sort_values(by='streetArea_prob' , axis = 1, ascending = False,inplace = True)
    # the result is the multipliciton of area and street probabilities
    # print(df_mat1)
    return df_mat1
def cost_classifier (Area1,street1,cost1):
    result1 = streets_classifier(Area1,street1,cost1)
    # classifing cost 5
    if cost1 == 5:
        for i in result1.iloc[4:,0:]:
            for j in result1.columns:
                if result1.loc['cost_prob_5',j] == 0 :
                    result1.drop(j, inplace=True, axis=1)
    # cost 4
    if cost1 == 4:
        for i in result1.iloc[3:,0:]:
            for j in result1.columns:
                if result1.loc['cost_prob_4',j] == 0 and result1.loc['cost_prob_5',j] == 0:
                    result1.drop(j, inplace=True, axis=1)
    # cost 3
    if cost1 == 3:
        for i in result1.iloc[2:,0:]:
            for j in result1.columns:
                if result1.loc['cost_prob_3',j] == 0 and result1.loc['cost_prob_4',j] == 0 and result1.loc['cost_prob_5',j] == 0 :
                    result1.drop(j, inplace=True, axis=1)
    # cost 2
    if cost1 == 2:
        for i in result1.iloc[1:,0:]:
            for j in result1.columns:
                if result1.loc['cost_prob_2',j] == 0 and result1.loc['cost_prob_3',j] == 0 and result1.loc['cost_prob_4',j] == 0 and result1.loc['cost_prob_5',j] == 0 :
                    result1.drop(j, inplace=True, axis=1)
    return result1.columns
def area_find(business):
    ww1= r1[[business]]
    ww2= r2[[business]]
    ww3= r3[[business]]
    ww4= r4[[business]]
    www=pd.concat([ww1,ww2,ww3,ww4], axis=0)
    www=www.sort_values(by=[business] ,ascending=False)
    xx=df5[[business]]
    m1=xx.loc['main_street',business]
    m2=xx.loc['main-substreet',business]
    m3=xx.loc['substreet',business]
    main=[]
    main_sub=[]
    sub=[]
    for i in www.index:
          if www.loc[i,business] == 0 :
                www=www.drop(i)
    for i in www.index:
          for j in www.columns:
                main.append(www.loc[i,j]*m1)
                main_sub.append(www.loc[i,j]*m2)
                sub.append(www.loc[i,j]*m3)
    www["Needs in main street"] = main
    www["Needs in main_sub street"] = main_sub
    www["Needs in sub street"] = sub
    www=www.drop([business], axis = 1)
    s_Areas = www.iloc[:,:1]
    Arro = s_Areas.index
    return Arro
def home(request):
    return render(request,'index.html')
def handle_data(request):
    area=int(request.GET['areas'])
    cost=int(request.GET['quantity'])
    street_cat = request.GET['street']
    dfc=pd.DataFrame(cost_classifier(area,street_cat,cost))
    html=dfc.to_numpy()
    return render(request, 'result.html', {'html': html })
def handle_data2(request):
      b_num=int(request.GET['business'])
      html=area_find(b_num)
      return render(request, 'result2.html', {'html': html })
def about(request):
    return render(request,'about.html')
def contact(request):
    return render(request,"contact.html")
