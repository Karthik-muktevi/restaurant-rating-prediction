import logging as lg
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


st.title('Restaurant Rating Prediction')

class data:
    def __init__(self):
        self.logger = lg

    def get_data(self,df):
        try:
            lg.info('Loading the dataframe')
            st.write('The dataframe sample')
            st.dataframe(df.sample(frac=0.005))
            lg.info('Loading dataframe succesfull')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def head(self,df):
        try:
            lg.info('Getting the first five rows and last five rows')
            st.write('The first five rows of the dataset')
            st.dataframe(df.head())
            lg.info('loading first five rows successfull')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def tail(self,df):
        try:
            lg.info('Getting the first five rows and last five rows')
            st.write('The last five rows of the dataset')
            st.dataframe(df.tail())
            lg.info('loading tail successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def null(self,df):
        try:
            lg.info('Getting the missing values')
            st.write('The number of rows is')
            st.write(df.shape[0])
            st.write('The number of columns is')
            st.write(df.shape[1])
            lg.info('getting shape values successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def missingvalues(self,df):
        try:
            lg.info('Getting the missing values')
            st.write('The missing values in the dataset')
            st.write(df.isnull().sum())
            lg.info('Getting the missing values successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def describe(self,df):
        try:
            lg.info('Describe')
            st.write('Details about the dataset features')
            st.write(df.describe())
            lg.info('Describe successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

select = st.selectbox(label='Select one of the following',options=['About','EDA','Predictions'])
if select=='About':
    submit = st.button(label='Submit')
    if submit:
        st.header('About Page')
        st.write(
            "Problem Statement: The main goal of this project is to perform extensive Exploratory Data Analysis(EDA) on the Zomato Dataset and build an appropriate Machine Learning Model that will help various Zomato Restaurants to predict their respective Ratings based on certain features.")
        df = pd.read_csv('zomato.csv')
        obj  = data()
        obj.get_data(df)
        obj.null(df)
        obj.head(df)
        obj.tail(df)
        obj.missingvalues(df)
        obj.describe(df)


class eda:

    def __init__(self):
        self.logger = lg
        lg.info('EDA')


    def Eda1(self,df):
        try:
            lg.info('eda1')
            st.write('Visualising the missing values')
            a1 = plt.figure(figsize=(10,5))
            sns.heatmap(df.isnull())
            st.pyplot(a1)
            st.write('features like rate, phone, location,rest_type,dish_liked,cuisines,approx_cost(for two people) have missing values')
            lg.info('eda1 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda2(self,df):
        try:
            lg.info('eda2')
            map_ = {'Yes': 1, 'No': 0}
            df['online_order'] = df['online_order'].map(map_)
            a2 = plt.figure(figsize=(8,5))
            df['online_order'].value_counts().plot(kind='barh')
            st.pyplot(a2)
            st.write('More number of the restaurants provide online ordering facility')
            lg.info('eda2 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda3(self,df):
        try:
            lg.info('eda3')
            map_ = {'Yes': 1, 'No': 0}
            df['book_table'] = df['book_table'].map(map_)
            a3 = plt.figure(figsize=(8,5))
            df['book_table'].value_counts().plot(kind='barh')
            st.pyplot(a3)
            st.write('Most of the restaurents do not have table booking facility')
            lg.info('eda3 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def Eda4(self,df):
        try:
            lg.info('eda4')
            a4 = plt.figure(figsize=(10, 5))
            df['name'].value_counts()[:20].plot(kind='barh')
            st.pyplot(a4)
            st.write('Cafe coffee day is the restaurent which has more branches followed by Onesta')
            lg.info('eda4 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def Eda5(self,df):
        try:
            lg.info('eda5')
            a5 = plt.figure(figsize=(10, 5))
            df['location'].value_counts()[:15].plot(kind='pie')
            st.pyplot(a5)
            st.write('BTM is the location which has more number of restaurents')
            lg.info('eda5 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

    def Eda6(self,df):
        try:
            lg.info('eda6')
            a6 = plt.figure(figsize=(10, 5))
            df['rest_type'].value_counts()[:20].plot(kind='barh')
            st.pyplot(a6)
            st.write('Quick bites and casual dining are the popular restaurent types')
            lg.info('eda6 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda7(self,df):
        try:
            lg.info('eda7')
            a7 = plt.figure(figsize=(10, 5))
            df['dish_liked'].value_counts()[:20].plot(kind='barh')
            st.pyplot(a7)
            st.write('Biryani is the most liked food followed by friendly staff')
            lg.info('eda7 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda8(self,df):
        try:
            lg.info('eda8')
            a8 = plt.figure(figsize=(10, 5))
            df['listed_in(type)'].value_counts()[:20].plot(kind='barh')
            st.pyplot(a8)
            st.write(' Delievery stands out of the all options available followed by dine out pubs and bars,buffet,drinks and nightlife are least engaged')
            lg.info('eda8 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda9(self,df):
        try:
            lg.info('eda9')
            a9 = plt.figure(figsize=(10, 5))
            cuisines_offered = df['cuisines'].value_counts()[:20]
            sns.barplot(x=cuisines_offered, y=cuisines_offered.index, data=df)
            st.pyplot(a9)
            st.write('North Indian and North Indian Chinese are the top most cuisines being offered whereas beverages stand last')
            lg.info('eda9 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda10(self,df):
        try:
            lg.info('eda10')
            a10 = plt.figure(figsize=(10, 5))
            listed_incity = df['listed_in(city)'].value_counts()[:20]
            sns.barplot(x=listed_incity, y=listed_incity.index, data=df)
            st.pyplot(a10)
            st.write('BTM,Koramangala 7th block,5th block are the cities which contain most of the restaurants Kammanahali has least')
            lg.info('eda10 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda11(self,df):
        try:
            lg.info('eda11')
            a11 = plt.figure(figsize=(10, 5))
            sns.scatterplot(x='online_order', y='votes', data=df)
            st.pyplot(a11)
            st.write('There seems to be some outliers for online order and other with respect to votes')
            lg.info('eda11 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda12(self,df):
        try:
            lg.info('eda12')
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            a12 = plt.figure(figsize=(10, 5))
            sns.scatterplot(x='rate', y='votes', data=df, hue='listed_in(type)')
            st.pyplot(a12)
            lg.info('eda12 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda13(self,df):
        try:
            lg.info('eda13')
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            a13 = plt.figure(figsize=(10, 5))
            sns.scatterplot(y='online_order', x='rate', data=df)
            st.pyplot(a13)
            st.write('People have rated similarly for both online and offline order,which may imply rating depends on quality of food')
            lg.info('eda13 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda14(self,df):
        try:
            lg.info('eda14')
            a14 = plt.figure(figsize=(10, 5))
            sns.barplot(x='listed_in(type)', y='votes', data=df)
            st.pyplot(a14)
            st.write('Drinks and nightlife was voted the highest')
            lg.info('eda14 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda15(self,df):
        try:
            lg.info('eda15')
            a15 = plt.figure(figsize=(10, 5))
            sns.barplot(y='listed_in(city)', x='votes', data=df)
            st.pyplot(a15)
            st.write('Indiranagar and oldairport road are the places with highest votes')
            lg.info('eda15 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda16(self,df):
        try:
            lg.info('eda16')
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            a16 = plt.figure(figsize=(10, 5))
            sns.scatterplot(x='listed_in(type)', y='rate', data=df)
            st.pyplot(a16)
            st.write('Almost all types have equal spread of ratings while drinks and nightlife has maximum as well as minimum rating')
            lg.info('eda16 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))
    def Eda17(self,df):
        try:
            lg.info('eda17')
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            a17 = plt.figure(figsize=(10, 5))
            sns.barplot(y='listed_in(city)', x='rate', data=df)
            st.pyplot(a17)
            st.write('Brigade road MG road,church street are some of the place where people ratings are high')
            lg.info('eda17 successful')
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

if select=='EDA':
    st.header('EDA')
    submit = st.button(label='EDA')
    if submit:
        df = pd.read_csv('zomato.csv')
        obj1 = eda()
        obj1.Eda1(df)
        obj1.Eda2(df)
        obj1.Eda3(df)
        obj1.Eda4(df)
        obj1.Eda5(df)
        obj1.Eda6(df)
        obj1.Eda7(df)
        obj1.Eda8(df)
        obj1.Eda9(df)
        obj1.Eda10(df)
        obj1.Eda11(df)
        obj1.Eda12(df)
        obj1.Eda13(df)
        obj1.Eda14(df)
        obj1.Eda15(df)
        obj1.Eda16(df)
        obj1.Eda17(df)


class preprocessing:
    def __init__(self):
        self.logger = lg

    def preprocess(self,df):
        try:
            lg.info('preprocessing')
            df.drop(['url', 'address', 'name', 'phone', 'reviews_list', 'menu_item', 'listed_in(city)', 'dish_liked'],
                axis=1, inplace=True)
            df['rate'].replace('NEW', np.NaN, inplace=True)
            df['rate'].replace('-', np.NaN, inplace=True)
            df['rate'] = df['rate'].astype(str)
            df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
            df['rate'] = df['rate'].apply(lambda x: float(x))
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(lambda x: x.replace(',', '.'))
            df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)
            df.rename(columns={'approx_cost(for two people)': 'approx_cost'}, inplace=True)
            df['rate'].fillna(df['rate'].mean(), inplace=True)
            df['location'].fillna(df['rate'].mode()[0], inplace=True)
            map_ = {'Yes': 1, 'No': 0}
            df['online_order'] = df['online_order'].map(map_)
            df['book_table'] = df['book_table'].map(map_)
            type_ = {'Buffet': 1, 'Cafes': 2, 'Delivery': 3, 'Desserts': 4, 'Dine-out': 5, 'Drinks & nightlife': 6,
                 'Pubs and bars': 7}
            df['listed_in(type)'] = df['listed_in(type)'].map(type_)
            df.rename(columns={'listed_in(type)':'listed_intype'},inplace=True)
            df.dropna(inplace=True)
            le = LabelEncoder()
            df.location = le.fit_transform(df.location)
            df.rest_type = le.fit_transform(df.rest_type)
            df.cuisines = le.fit_transform(df.cuisines)
            lg.info('train preprocessing successful')
            return df
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

obj2 = preprocessing()
zom = pd.read_csv('zomato.csv')

df = obj2.preprocess(zom)

class split:
    def __init__(self):
        self.logger = lg
    def train_split(self,df):
        try:
            lg.info('train split')
            x = df[['online_order','book_table','votes','location','rest_type','cuisines','approx_cost','listed_intype']]
            y = df['rate']
            lg.info('split successful')
            print(type(x))
            return [x,y]
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

obj3 = split()
x1 = obj3.train_split(df)
#y2 = obj3.train_split(df)
x=x1[0]
y=x1[1]
class prediction:
    def __init__(self):
        self.logger = lg
    def predict(self,x,y,test):
        try:
            lg.info('Random Forest algorithm')
            rf = RandomForestRegressor()
            rf.fit(x,y)
            y_pred = rf.predict(test)
            lg.info('training successful')
            return y_pred
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

class test_preprocess:
    def __init__(self):
        self.logger = lg
    def test_preprocess1(self,test_df):
        try:
            lg.info('test preprocessing')
            #test_df['approx_cost'] = test_df['approx_cost'].astype(str)
            #test_df['approx_cost'] = test_df['approx_cost'].apply(lambda x: x.replace(',', '.'))
            #test_df['approx_cost'] = test_df['approx_cost'].astype(float)
            map1 = {'Yes': 1, 'No': 0}
            test_df.online_order = test_df.online_order.map(map1)
            test_df.book_table = test_df.book_table.map(map1)
            type_ = {'Buffet': 1, 'Cafes': 2, 'Delivery': 3, 'Desserts': 4, 'Dine-out': 5, 'Drinks & nightlife': 6,
                 'Pubs and bars': 7}
            test_df.listed_intype = test_df.listed_intype.map(type_)
            le = LabelEncoder()
            test_df.location = le.fit_transform(test_df.location)
            test_df.rest_type = le.fit_transform(test_df.rest_type)
            test_df.cuisines = le.fit_transform(test_df.cuisines)
            lg.info('test preprocessing successful')
            return test_df
        except Exception as e:
            lg.error('Error')
            lg.exception(str(e))

if select=='Predictions':
    zom = pd.read_csv('zomato.csv')
    online_order = st.selectbox(label='online_order',options=['Yes','No'])
    book_table = st.selectbox(label='book_table',options=['Yes','No'])
    votes = st.number_input(label='votes')
    location = st.selectbox(label='location',options=zom.location.unique().tolist())
    rest_type = st.selectbox(label='rest_type',options=zom.rest_type.unique().tolist())
    cuisines = st.selectbox(label='cuisines',options=zom.cuisines.unique().tolist())
    approx_cost = st.number_input(label='approximate cost for two')
    listed_intype = st.selectbox(label='listed in type',options=zom['listed_in(type)'].unique().tolist())

    submit = st.button(label='Get predictions')
    if submit:
        test_df1 = pd.DataFrame([[online_order,book_table,votes,location,rest_type,cuisines,approx_cost,listed_intype]],columns=['online_order','book_table','votes','location','rest_type','cuisines','approx_cost','listed_intype'])
        #st.dataframe(test_df1)
        obj4 = test_preprocess()
        test_df = obj4.test_preprocess1(test_df1)
        obj5 = prediction()
        pred = obj5.predict(x,y,test_df)
        st.write('The predicted rating is {}'.format(pred))























