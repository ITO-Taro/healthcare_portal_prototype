from flask import *
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, VARCHAR, Integer, Column, Date, Float
import math, os
import numpy as np
import pandas as pd


PATH = os.getcwd()
os.chdir(PATH)

app = Flask(__name__)

app.config['SQLALCHERMY_DATABASE_URI'] = "sqlite:///health_care_portal.db"

# app.config["sqlite:///health_care_portal.db"] = 'postgresql+psycopg2://login:pass@localhost/flask_app'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# db.create_all()

# db.session.commit()

Base = declarative_base()

engine = create_engine("sqlite:///health_care_portal.db")

class MedCode(Base):
    __tablename__ = 'med_codes'

    id = Column(Integer, primary_key=True)
    code = Column(VARCHAR(5), nullable=False)
    description = Column(VARCHAR(50), nullable=False)
    category = Column(VARCHAR(30), nullable=False)

class Employee(Base):
    __tablename__ = "patient_accounts"

    id = Column(Integer, primary_key=True)
    emp_id = Column(VARCHAR(20), nullable=False)
    title = Column(VARCHAR(30), nullable=False)
    gender = Column(VARCHAR(10), nullable=False)
    last_name = Column(VARCHAR(30), nullable=False)
    first_name = Column(VARCHAR(30), nullable=False)
    salary = Column(Integer, nullable=False)
    city = Column(VARCHAR(20), nullable=False)
    state = Column(VARCHAR(10), nullable=False)

class Transactions(Base):
    __tablename__ = "patient_transactions"

    id = Column(Integer, primary_key=True)
    emp_id = Column(VARCHAR(20), nullable=False)
    trans_id = Column(VARCHAR(15), nullable=False)
    procedure_date = Column(Date, nullable=False)
    medical_code = Column(VARCHAR(10), nullable=False)
    procedure_price = Column(Float, nullable=False)

class Stats:
    '''
    attempting to develop ANOVA function
    '''

    # def __init__(self, data_x, data_y):
    #     self.data_x = data_x
    #     self.data_y = data_y
    #     self.x_mean = self.mean(self.data_x)
    #     self.y_mean = self.mean(self.data_y)
    #     self.sp_xy = sum([(grade - self.x_mean)*(salary - self.y_mean) for grade in self.data_x for salary in self.data_y])
    #     self.ss_x = sum([(grade - self.x_mean)**2 for grade in self.data_x])
    #     self.slope = self.sp_xy/self.ss_x
    #     self.y_intcpt = self.y_mean - self.slope*self.x_mean
    #     self.tss = sum([(salary - self.y_mean)**2 for salary in self.data_y])
    #     self.rss = sum([(self.y_intcpt + self.slope * salary) for salary in data_y])
    #     self.sse = sum([(salary - (self.y_intcpt + self.slope * salary))**2 for salary in data_y])

    def calc_mean(self, data):
        return sum(data)/len(data)
    
    def find_modes(self, var_x):
        data, modes = {}, {}
        var_x.sort()
        cnt = 1
        for n in range(1, len(var_x)):
            if var_x[n] == var_x[n-1]:
                cnt += 1
            else:
                data[var_x[n-1]] = cnt
                cnt = 1
        for k, v in data.items():
            if v == max(data.values()) and v > 1:
                modes[k] = v
        return modes

    def find_median(self, var_x):
        var_x.sort()
        num = len(var_x)
        res = 0
        if num%2 == 0:
            res = (var_x[int(num/2)]+var_x[int(num/2)-1])/2
        else:
            res = var_x[int((num-1)/2)]
        return res

    def calc_SD(self, var_x):
        cnt, avg = len(var_x), Stats().calc_mean(var_x)
        return math.sqrt(sum([(i - avg)**2 for i in var_x])/(cnt-1))
    
    def basic_stats(self, data_list):
        '''
        Calculates and returns basic stats values as a dict
        '''
        stats = dict()
        data = pd.Series(data_list)
        data = data.apply(lambda x: float(x))
        stats["sample_size"] = data.count()
        stats["mean"] = round(data.mean(), 4)
        stats["median"] = round(data.median(), 4)
        stats["modes"] = Stats().find_modes(data.to_numpy().astype(np.float64))
        stats["sd"] = round(data.std(), 4)
        return stats


Base.metadata.bind = engine
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
session = DBSession()