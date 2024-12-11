import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from functools import partial
import random
import time
import sqlite3
import pickle
from fpdf import FPDF
import re

def reset_state():
    st.session_state.store_number = 1
    st.session_state.today_date = datetime.today()
    st.session_state.forecast_date = datetime.today() 

def validate_dataframe(df, expected_columns):
    if df.shape[1] != len(expected_columns):
        return False
    for col, dtype in expected_columns.items():
        if col not in df.columns:
            return False
        if df[col].dtype != dtype:
            return False
    return True

def load_model_from_db():
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()

    cursor.execute('SELECT model FROM models WHERE id = ?', (1,))
    model_blob = cursor.fetchone()[0]

    model = pickle.loads(model_blob)
    conn.close()
    return model

def process_store_data(df, target_store, target_date):
    train_dfs = []
    unique_stores = df['Store'].unique()
    target_store_processed = False

    for store in unique_stores:
        store_df = df[df['Store'] == store]
        eligible_dates = store_df[store_df['Date'] < target_date]

        if len(eligible_dates)<135:#обучение модели+оптимизация гиперпараметра
            continue
        idx = eligible_dates.index[-1]
        train_df = store_df.loc[idx - 134:]
        if (target_date-train_df.iloc[0]['Date']).days>953:
            continue
        train_dfs.append(train_df)
        if store == target_store:
            target_store_processed = True

    if not target_store_processed:
        return False, None

    final_train_df = pd.concat(train_dfs).reset_index(drop=True)
    return True, final_train_df

def generate_pdf(lines):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    for line in lines:
        pdf.multi_cell(0, 10, txt=line, align='L')
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

def extract_number(company_name):
    match = re.search(r'\d+', company_name)
    return int(match.group()) if match else float('inf')

#Прогноз
#каждый объект создается для одного магазина
class ExpSmooth:
  #data-история для одного магазина
  #L=52
  #alpha, beta, gamma-гиперпараметры
  #m-горизонт прогнозирования
  #result-результат прогнозирования
  #S-уровень b-тренд I сезонность

  def __init__(self,data,alpha, beta, gamma,m,L=52):
    self.data=data
    self.L=L
    self.alpha=alpha
    self.beta=beta
    self.gamma=gamma
    self.m=m
  def initial_b(self):
    b=0.0
    for i in range(self.L):
      b+=float(self.data[i+self.L]-self.data[i])
    return b/float(self.L**2)
  def initial_I(self):
    I=[]
    A1=0.0
    A2=0.0
    for i in range(self.L):
      A1+=self.data[i]
      A2+=self.data[i+self.L]
    A1=A1/float(52)
    A2=A2/float(52)
    for i in range(self.L):
      I.append((self.data[i]/A1+self.data[i+self.L]/A2)/2)
    I.append(self.beta*self.data[self.L]/self.data[self.L]+(1-self.beta)*I[0])
    return I
  def smoothing(self):
    self.result=[]#0 index->2L
    S=0.0
    S_prev=self.data[self.L]
    b=self.initial_b()
    I=self.initial_I()
    #train
    for i in range(self.L+1,self.L*2):
      S=self.alpha*self.data[i]/I[i-self.L]+(1-self.alpha)*(S_prev+b)
      b=self.gamma*(S-S_prev)+(1-self.gamma)*b
      I.append(self.beta*self.data[i]/S+(1-self.beta)*I[i-self.L])
      S_prev=S
    #test
    for i in range(self.m):
      self.result.append(I[self.L+i]*(S+self.m*b))

def mean_errors(x, train_df):
  errors=[]
  alpha, beta, gamma=x
  sales_matrix = train_df.pivot(index='Store', columns='Date', values='Weekly_Sales')
  sales_array = sales_matrix.to_numpy()
  for s in sales_array:
    validation_len=31
    model=ExpSmooth(s,alpha=alpha,beta=beta,gamma=gamma,m=validation_len)
    model.smoothing()
    predictions=model.result
    actual=s[model.L*2:model.L*2+validation_len]
    error=mean_squared_error(predictions, actual)
    errors.append(error)
  return np.mean(np.array(errors))

def hyperparm_selection(train_df):
  x = [0, 0, 0]
  our_er = partial(mean_errors, train_df=train_df)
  optimal=minimize(our_er,x0=x,method="TNC", bounds = ((0, 1), (0, 1), (0, 1)))
  alpha_final, beta_final, gamma_final = optimal.x
  return alpha_final, beta_final, gamma_final

def make_graphic_comparison(results, test_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_df['Date'], test_df['Weekly_Sales'], label='Реальные результаты', marker='o')
    ax.plot(test_df['Date'], results, label='Предсказанный результат', marker='x')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Продажи')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def make_graphic(start_date, num_weeks, results):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')  # Преобразуем Timestamp в строку
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(weeks=i) for i in range(num_weeks)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, results, marker='o', linestyle='-', color='green', label='Предсказанный результат')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    ax.set_xlabel('Дата')
    ax.set_ylabel('Продажи')
    ax.legend()
    st.pyplot(fig)

def metrics(test_df,results,count_weeks_prognosing):
  true_values=test_df['Weekly_Sales'].tolist()
  r2 = r2_score(true_values, results)
  mape=0.0
  rmse=0.0
  for i in range(count_weeks_prognosing):
    mape+=abs(true_values[i]-results[i])/true_values[i]
    rmse+=(true_values[i]-results[i])**2
  mape=mape/count_weeks_prognosing*100
  rmse=rmse/count_weeks_prognosing
  rmse=rmse**0.5
  return r2,mape,rmse

#Классификация
def generate_feature(size, allow_zero=False):
    if allow_zero:
        return np.random.randint(0, 11, size)
    else:
        return np.random.randint(1, 11, size)

def generate_delivery_time(delivery_on_time):
    delivery_time = np.where(delivery_on_time == 0, np.nan,
                             33 - 3 * delivery_on_time)
    return delivery_time

def generate_production_time(lead_time):
    return 77 - 7 * lead_time

def determine_mean(row):
  score = (row['Quality M']+ row['Quality D'] + row['Delivery on time'] + row['Material price'] +
             row['Information sharing'] + row['After sales service'] +
             row['Lead time'] + row['Quantity discount'] +
             row['Occupational health and safety system M'] +
             row['Occupational health and safety system D'] +
             row['Transportation cost']) / 11
  return score

def determine_supplier_class(row):
    score = (row['Quality M']+ row['Quality D'] + row['Delivery on time'] + row['Material price'] +
             row['Information sharing'] + row['After sales service'] +
             row['Lead time'] + row['Quantity discount'] +
             row['Occupational health and safety system M'] +
             row['Occupational health and safety system D'] +
             row['Transportation cost']) / 11

    random_adjustment = np.random.normal(0, 0.5)  #(мат.ожидание, дисперсия)
    adjusted_score = score + random_adjustment

    if adjusted_score >= 7:
        return 'excellent'
    elif adjusted_score >= 5:
        return 'satisfactory'
    else:
        return 'bad'
#генерация поставщиков
def generate_manufacturers(data_size):
  np.random.seed(int(time.time()))
  data = {
      'Quality M': generate_feature(data_size),
      'Material price': generate_feature(data_size),
      'Information sharing': generate_feature(data_size),
      'After sales service': generate_feature(data_size, allow_zero=True),
      'Lead time': generate_feature(data_size),
      'Quantity discount': generate_feature(data_size, allow_zero=True),
      'Occupational health and safety system M': generate_feature(data_size)
  }
  data['Production time'] = generate_production_time(data['Lead time'])
  df = pd.DataFrame(data)
  return df
#генерация доставщиков
def generate_delivery(data_size):
  np.random.seed(int(time.time()))
  data = {
      'Quality D': generate_feature(data_size),
      'Delivery on time': generate_feature(data_size, allow_zero=True),
      'Occupational health and safety system D': generate_feature(data_size),
      'Transportation cost': generate_feature(data_size),
  }
  data['Delivery time'] = generate_delivery_time(data['Delivery on time'])
  df = pd.DataFrame(data)
  return df
def balanced_generate_dataset(manufacturers_count, delivery_count):
    np.random.seed(int(time.time()))
    manufacturers = generate_manufacturers(manufacturers_count)
    delivery = generate_delivery(delivery_count)
    dataset = manufacturers.merge(delivery, how='cross')
    dataset['Supplier Class'] = dataset.apply(determine_supplier_class, axis=1)
    dataset['Mean'] = dataset.apply(determine_mean, axis=1)
    class_counts = dataset['Supplier Class'].value_counts()
    min_class_count = class_counts.min()
    balanced_dataset = pd.concat([
        dataset[dataset['Supplier Class'] == 'excellent'].sample(min_class_count, random_state=1),
        dataset[dataset['Supplier Class'] == 'satisfactory'].sample(min_class_count, random_state=1),
        dataset[dataset['Supplier Class'] == 'bad'].sample(min_class_count, random_state=1)
    ])
    return balanced_dataset

#метод случайных подпространств
def bagging(k,df):
  features=['Quality M','Quality D', 'Delivery on time', 'Material price', 'Information sharing',
            'After sales service', 'Lead time','Quantity discount', 'Occupational health and safety system M',
            'Occupational health and safety system D', 'Transportation cost']
  random_strings = random.sample(features, k)
  df_bagging=pd.DataFrame()
  for st in random_strings:
      df_bagging[st]=df[st]
  df_bagging['Supplier Class']=df['Supplier Class']
  return df_bagging

def gini(P):
  er=0
  for p in P:
    er+=p**2
  er=er*(-1)
  er+=1
  return er

class Decision_node:
  def train(self, train_df):
    self.left=None
    self.right=None
    unique_values={}
    for feat in train_df.columns:
      if feat=="Supplier Class":
        continue
      uniq_val=sorted(train_df[feat].unique())
      unique_values[feat]=uniq_val
    #выбираем фичу
    feat_dist={}
    for feat in unique_values:
      arr=unique_values[feat]
      if len(arr)==1:
        average_array=arr
      else:
        average_array = [(arr[i] + arr[i + 1]) / 2 for i in range(len(arr) - 1)]
      #проходимся по всем средним значениям
      average_dist={}
      for item in average_array:
        left_part={}
        right_part={}
        #деалем распределение
        for i in range(len(train_df)):
          class_name=train_df.iloc[i]['Supplier Class']
          if(train_df.iloc[i][feat]<=item):
            if class_name in left_part:
              left_part[class_name]+=1
            else:
              left_part[class_name]=1
          else:
            if class_name in right_part:
              right_part[class_name]+=1
            else:
              right_part[class_name]=1
        #левая часть
        p_left=[]
        left_count=0
        for sup_class in left_part:
          left_count+=left_part[sup_class]
        for sup_class in left_part:
          p_left.append(left_part[sup_class]/left_count)
        #правая часть
        p_right=[]
        right_count=0
        for sup_class in right_part:
          right_count+=right_part[sup_class]
        for sup_class in right_part:
          p_right.append(right_part[sup_class]/right_count)
        #считаем ошибку G для этого значения
        J=(left_count/len(train_df))*gini(p_left)+(right_count/len(train_df))*gini(p_right)
        average_dist[item]=J
      min_pair=min(average_dist.items(),key=lambda x:x[1])#выбрали разбиение в фиче с минимальной ошибкой
      key, value = min_pair
      feat_dist[(feat,key)]=value#записали для фичи разбиение с минимальной ошибкой
    min_feat_val=min(feat_dist.items(),key=lambda x:x[1])#выбрали фичу со значением наилучшего разбиения
    (feat,key),value=min_feat_val
    self.feauture=feat
    self.value=key
    self.J=value
  def check(self, train_df,phi):#phi=0.0195-лучший
    node_dist={}
    for i in range(len(train_df)):
      class_name=train_df.iloc[i]['Supplier Class']
      if class_name in node_dist:
        node_dist[class_name]+=1
      else:
        node_dist[class_name]=1
    p_current=[]
    for cl in node_dist:
      p_current.append(node_dist[cl]/len(train_df))
    delta_G=gini(p_current)-self.J
    if(delta_G>=phi):
      return True
    else:
      return False

class Leaf:
  def __init__(self,train_df):#train_df разбитый left/right
    pr_ds={}
    for i in range(len(train_df)):
      class_name=train_df.iloc[i]['Supplier Class']
      if class_name in pr_ds:
        pr_ds[class_name]+=1
      else:
        pr_ds[class_name]=1
    for pr in pr_ds:
      pr_ds[pr]=pr_ds[pr]/len(train_df)
    self.probability_distribution=pr_ds

class Tree:
  def __init__(self,train_df,phi):
    self.root_node=self.go(train_df,phi)

  def go(self,train_df,phi):
    if len(train_df)==1:
      A=Leaf(train_df)
      return A
    A=Decision_node()
    A.train(train_df)
    if(A.check(train_df,phi)):
      left_df=pd.DataFrame(columns=train_df.columns)
      right_df=pd.DataFrame(columns=train_df.columns)
      for i in range(len(train_df)):
        if(train_df.iloc[i][A.feauture]<=A.value):
          left_df.loc[len(left_df)]=train_df.loc[i]
        else:
          right_df.loc[len(right_df)]=train_df.loc[i]
      A.left=self.go(left_df,phi)
      A.right=self.go(right_df,phi)
    else:
      A=Leaf(train_df)
    return A

  def classify(self,sample):#sample строка из dataframe
    current_node=self.root_node
    while type(current_node)!=Leaf:
        if sample[current_node.feauture]<=current_node.value:
          current_node=current_node.left
        else:
          current_node=current_node.right
    return current_node.probability_distribution

class Random_Forest:
  def __init__(self, trees_count, dataset,phi):
    trees_array=[]
    for i in range(trees_count):
      train_df=bagging(4,dataset)
      our_tree=Tree(train_df,phi)
      trees_array.append(our_tree)
    self.forest=trees_array

  def predict(self,sample):#sample это словарь с фичами и значениями
    final_distribution={}
    for tr in self.forest:
      dis=tr.classify(sample)
      for d in dis:
        if d in final_distribution:
          final_distribution[d]+=dis[d]
        else:
          final_distribution[d]=dis[d]
    for f in final_distribution:
      final_distribution[f]=final_distribution[f]/len(self.forest)
    max_likely_class=max(final_distribution,key=final_distribution.get)
    max_prob=final_distribution[max_likely_class]
    # print(max_prob)
    return max_likely_class
  def F1(self,validation_df, need_class):
    TP=0
    FP=0
    FN=0
    for i in range(len(validation_df)):
      predict_class=self.predict(validation_df.loc[i])
      if predict_class==need_class and validation_df.loc[i]['Supplier Class']==need_class:
        TP+=1
      elif predict_class==need_class and validation_df.loc[i]['Supplier Class']!=need_class:
        FP+=1
      elif predict_class!=need_class and validation_df.loc[i]['Supplier Class']==need_class:
        FN+=1
    if TP==0:
      F1=0
    else:
      precission=TP/(TP+FP)
      recall=TP/(TP+FN)
      F1=2*precission*recall/(precission+recall)
    print('TP=', TP, ' FP=',FP,' FN=',FN,' F1=',F1)
    return F1