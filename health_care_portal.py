from flask import *
from models import *
from charts import *
import io, base64, os
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sqlalchemy import create_engine, func, and_, desc, asc, extract, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

'''
Contains all functions used/called in emp_analysis_project.py (app file)
'''

PATH = os.getcwd()
os.chdir(PATH)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///health_care_portal.db"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

db.create_all()

db.session.commit()

Base = declarative_base()

engine = create_engine("sqlite:///health_care_portal.db?check_same_thread=False")
Base.metadata.bind = engine
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
session = DBSession()

TODAY = pd.Timestamp.today().date()


class HealthCarePortal():
    LIMIT = None
    GENDERS = ["F", "M"]

    def __init__(self):
        self.description = ProjectDescription.project_description("health_care_portal_description.txt")

        if not self.__is_empty(MedCode):
            self.med_code_table()
        if not self.__is_empty(Employee):
            self.emp_table()
        if not self.__is_empty(Transactions):
            self.transactions_table()
        

    def med_code_table(self):
        df = pd.read_excel("2021_medical_codes.xlsx", index_col=False, usecols=["CODE","DESCRIPTION","CATEGORY"])
        for col in ["CODE", "DESCRIPTION", "CATEGORY"]:
            if col == "CODE":
                df[col] = df[col].apply(lambda x: str(x).strip(" "))
            else:
                df[col] = df[col].apply(lambda x: "".join([i for i in x.lower() if ord(i) in range(97, 123) or ord(i) in range(47, 58) or i == " "]).title())
        for num in df.index:
            session.add(MedCode(code = df.loc[num, "CODE"], description = df.loc[num, "DESCRIPTION"], category = df.loc[num, "CATEGORY"]))
            session.commit()

    def emp_table(self):
        df = pd.read_csv("patient_accounts.txt", index_col=False, names=["emp_id", "title", "gender", "last_name", "first_name", "salary", "city", "state"])
        df = df.fillna("")
        for col in df.columns:
            df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i.lower()) in range(97, 123) or ord(i) in range(48, 58) or i == "%" or i == " " or i == "-"]))
            if col == "salary":
                df[col] = df[col].astype(int)
            elif col == "city":
                df[col] = df[col].apply(lambda x: x.replace("%", " ").title())
            elif col == "first_name":
                df[col] = df[col].apply(lambda x: x.strip(" ").title())

        df.to_sql("patient_accounts", con=engine, if_exists="append", index="id")

    def transactions_table(self):
        df = pd.read_csv("patient_transactions.csv", names=["emp_id", "trans_id", "procedure_date", "medical_code", "procedure_price"], converters={"procedure_price": float}, parse_dates=["procedure_date"])

        for col in ["emp_id", "trans_id", "medical_code"]:
            df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i) in range(48, 58) or ord(i.lower()) in range(97, 123) or i == "-"]))
        df.to_sql("patient_transactions", con=engine, if_exists="append", index="id")
    
    def med_code_search(self, input):
        res = dict()
        data = session.query(MedCode).filter(MedCode.code == input)
        if data.first():
            res['code'], res['description'], res['category'] = data[0].code, data[0].description, data[0].category
        else:
            return render_template("med_codes.html", message="Invalid Medical Code")
        
        return render_template("med_codes_result.html", code=res['code'], description=res['description'], category=res['category'])

    
    def med_descript_search(self, input):
        if len(input) > 1:
            data = session.query(MedCode).filter(MedCode.description.like(f"%{input}%"))
            if data.first():
                data = data.all()
                result = f"{len(data)} records found matching '{input}'"
                return render_template("med_descript_result.html", data=data, result=result)
            else:
                return render_template("med_descript.html", message="Invalid Key Word")
        else:
            return render_template("med_descript.html", message="Invalid Key Word")

    def emp_search_by_id(self, input):
        if len(input) < 2:
            res = "Invalid Employee ID"
            return render_template("emp_search.html", message=res)
        else:
            data = session.query(Employee).filter(Employee.emp_id == input)
            if data.first():
                data = data.all()
                res = f"{len(data)} record(s) found matching '{input}'"
                return render_template("emp_search_result.html", data=data, result=res)
            else:
                res = f"No Employee Found For: {input}"
                return render_template("emp_search.html", message=res)
            
    def emp_search_by_last(self, input):
        if len(input) < 2:
            res = "Invalid Last Name"
            return render_template("emp_search_last.html", message=res)
        
        else:
            data = session.query(Employee).filter(Employee.last_name.like(f"%{input}%"))
            if data.first():
                data = data.limit(self.LIMIT).all() if self.LIMIT else data.all()
                res = f"{len(data)} record(s) found matching '{input}'"
                return render_template("emp_search_last_result.html", data=data, result=res)
            else:
                res = "Invalid Last Name"
                return render_template("emp_search_last.html", message=res)
    
    def emp_search_salary_range(self, input_min, input_max):
        if len(input_min) < 2 or len(input_max) < 2:
            res = "Invlaid Range"
            return render_template("emp_search_salary_range.html", message=res)
        else:
            start = int(input_min+"000") if len(input_min) in range(2,4) else int(input_min)
            end = int(input_max+"000") if len(input_max) in range(2,4) else int(input_max)
            data = session.query(Employee).filter(and_(Employee.salary >= start, Employee.salary < end))
            if data.first():
                data = data.order_by(desc(Employee.salary)).all()
                res = f"{len(data)} employee(s) found for salaries between ${start} and ${end}"
                return render_template("emp_search_salary_range_result.html", data=data, result=res)
            else:
                res = "Invalid Range"
                return render_template("emp_search_salary_range.html", message=res)

    def emp_portal(self, input_id):
        '''
        param: a complete employee id which == 10 in length
        '''
        if len(input_id) != 10:
            res = "Invalid Employee ID"
            return render_template("employee_portal.html", message=res)
        else:
            data_emp = session.query(Employee).filter(Employee.emp_id == input_id)
            data_transaction = data2 = session.query(Transactions).filter(Transactions.emp_id == input_id)
            if data_emp.first():
                count = data_transaction.count()
                total = session.query(func.sum(Transactions.procedure_price)).filter(Transactions.emp_id == input_id)[0][0]
                data_emp = data_emp.all()[0]
                data_transaction = data_transaction.order_by(asc(Transactions.procedure_date)).all()
                res = f"{len(data_transaction)} transaction record(s) found for employee {input_id}"

                return render_template("employee_portal_result.html", data_emp=data_emp, data_transaction=data_transaction, result=res, count=count, total=total)
    
            else:
                return render_template("employee_portal.html", message="Invalid Employee ID")
    
    def hr_portal(self, input_info):
        try:
            data_med = session.query(MedCode).filter(MedCode.code == input_info).limit(1)[0]
        except:
            msg = "Invalid Employee ID or Medical Code"
            return render_template("hr_portal_med_code.html", message=msg)

        res = dict()
        data_emp_and_transaction = session.query(Employee, Transactions).filter(and_(Transactions.medical_code == input_info, Employee.emp_id == Transactions.emp_id)).order_by(Transactions.procedure_date)
        
        if data_emp_and_transaction.first():
            data_emp_and_transaction = data_emp_and_transaction.limit(15)
            for gender in self.GENDERS:
                emp_gender = "Female" if gender == "F" else "Male"
                mean = session.query(func.avg(Transactions.procedure_price)).filter(and_(Transactions.medical_code == input_info, Employee.emp_id == Transactions.emp_id, Employee.gender == gender[0])).all()[0][0]
                res[gender] = f"Average Cost per {emp_gender} Patient: ${round(mean, 2)}"
            data_emp_and_transaction = data_emp_and_transaction.all()
            med_code = f"{data_med.code}: {data_med.description}"
        
            return render_template("hr_portal_med_code_result.html", data=data_emp_and_transaction, result=med_code, male_avg=res["M"], female_avg=res["F"])
            
        else:
            return render_template("hr_portal_med_code.html", message="Invalid Medical Code")
    
    def hr_summary(self):
        res = dict()
        years = session.query(extract("year", Transactions.procedure_date)).distinct().order_by(extract("year", Transactions.procedure_date)).all()
        for num in range(len(years)):    
            data = session.query(func.count(Employee.emp_id.distinct()), func.count(Transactions.trans_id), func.sum(Transactions.procedure_price), func.avg(Transactions.procedure_price)).filter(and_(extract('year', Transactions.procedure_date) == years[num][0]), Employee.emp_id == Transactions.emp_id).all()[0]
            all_price = session.query(Transactions.procedure_price).filter(and_(extract('year', Transactions.procedure_date) == years[num][0])).order_by(asc(Transactions.procedure_price)).all()
            prices = [all_price[num][0] for num in range(len(all_price))]
            median = Stats().find_median(prices)
            modes = Stats().find_modes(prices)
            sd = round(np.array(prices).std(), 2)
            res[years[num][0]] = [data[0], data[1], data[2], round(data[3], 2), median, modes, sd]
        
        return res

    def hr_benefit_cost(self):
        res = dict()
        titles = session.query(Employee.title).distinct().order_by(Employee.title).all()
        years = session.query(extract("year", Transactions.procedure_date)).distinct().order_by(extract("year", Transactions.procedure_date)).all()
        for title_num in range(len(titles)):
            lst = []
            for year_num in range(len(years)):
                data_by_year = dict()
                data = session.query(func.count(Employee.emp_id.distinct()), func.count(Transactions.trans_id), func.sum(Transactions.procedure_price), func.avg(Transactions.procedure_price)).filter(and_(extract('year', Transactions.procedure_date) == years[year_num][0]), Employee.emp_id == Transactions.emp_id, Employee.title == titles[title_num][0]).all()[0]
                all_price = session.query(Transactions.procedure_price).filter(and_(extract('year', Transactions.procedure_date) == years[year_num][0])).order_by(asc(Transactions.procedure_price)).all()
                prices = [all_price[n][0] for n in range(len(all_price))]
                median = Stats().find_median(prices)
                modes = Stats().find_modes(prices)
                sd = round(np.array(prices).std(), 2)
                data_by_year[years[year_num][0]] = [data[0], data[1], data[2], round(data[3], 2), median, modes, sd]
                lst.append(data_by_year)
            res[titles[title_num][0]] = lst
        return res

    def med_proc_analysis_sql(self):
        res = dict()
        conn = engine.connect()
        statement_by_count = text("SELECT mc.code, mc.description, pt.cnt as 'Total_Count' FROM med_codes as mc JOIN (SELECT medical_code, Count(medical_code) as 'cnt' FROM patient_transactions GROUP by medical_code ORDER by cnt DESC LIMIT(10)) as pt ON mc.code = pt.medical_code GROUP by mc.code ORDER by Total_Count DESC;")
        top10count = conn.execute(statement_by_count).all()
        res['top10count'] = top10count
        
        statement_by_cost = text("""SELECT mc.code, mc.description, pt.Total_Cost as "Total_Procedure_Cost" FROM med_codes as mc JOIN (SELECT medical_code, Sum(procedure_price) as 'Total_Cost' FROM patient_transactions GROUP by medical_code ORDER by Total_Cost DESC LIMIT(10)) as pt ON mc.code = pt.medical_code GROUP by mc.code ORDER by Total_Procedure_Cost DESC;""")
        top10cost = conn.execute(statement_by_cost).all()
        res['top10cost'] = top10cost
        
        statement_by_emp_cost = text("SELECT pa.emp_id, pa.last_name, pa.first_name, pa.gender, pt.total_cost as 'Total_Medical_Cost' FROM patient_accounts as pa JOIN(SELECT emp_id, SUM(procedure_price) as total_cost FROM patient_transactions GROUP by emp_id ORDER by total_cost DESC LIMIT(10)) as pt ON pa.emp_id = pt.emp_id GROUP by pa.emp_id ORDER by Total_Medical_Cost DESC")
        top10emp = conn.execute(statement_by_emp_cost).all()
        res['top10emp'] = top10emp
        
        conn.close()

        return res
    
    def pd_data(self, file, indx, columns):
        df = pd.read_csv(file, index_col=indx, names=columns)
        df = df.fillna("")
        return df

    def chart_hist(self, data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res

    def chart_cumulative(self, data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin, cumulative=True, histtype="step")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res

    def project_description(self, file):
        texts = []
        with open(file, 'r') as file:
            line = file.readline()
            cnt = 0
            while line:
                if len(line) > 5:
                    texts.append(line)
                line = file.readline()
                cnt += 1
        return texts
    
    def __pd_data(self, x, y, z):
        df = pd.read_csv(x, index_col=y, names=z)
        df = df.fillna("")
        return df

    def __is_empty(self, table):
        return session.query(table).first()
    
    def __is_not_empty(self, table):
        if session.query(table).first():
            return True
        return False
    
    def __all_med_codes(self):
        code_dict = dict()
        for gender in self.GENDERS:
            codes = session.query(Transactions.medical_code).filter(and_(Transactions.emp_id == Employee.emp_id, Employee.gender == gender)).distinct().order_by(Transactions.medical_code).all()
            code_dict[gender] = codes
        codes = sorted(set(code_dict["F"]) & set(code_dict["M"]))
        res = [i[0] for i in codes]
        return res
    
    def __med_cost_analysis_result(self, code_descript, res_dict, proc_price_dict):
        for code in self.med_codes:
            res_dict[code] = dict()
            description = code_descript[code]
            
            female = self.__filter_by_med_code(self.female_emp, code)
            male = self.__filter_by_med_code(self.male_emp, code)

            proc_price_dict['female'] = self.__get_proc_price(female)
            proc_price_dict['male'] = self.__get_proc_price(male)

            pricesM = [round(proc_price_dict['male'][num], 4) for num in range(len(proc_price_dict['male']))]
            pricesF = [round(proc_price_dict['female'][num], 4) for num in range(len(proc_price_dict['female']))]
            meanM, meanF = round(Stats().calc_mean(pricesM),2), round(Stats().calc_mean(pricesF),2)
            medianM, medianF = round(Stats().find_median(pricesM),4), round(Stats().find_median(pricesF),4)
            modesM, modesF = Stats().find_modes(pricesM), Stats().find_modes(pricesF)
            sdM, sdF = round(Stats().calc_SD(pricesM),4), round(Stats().calc_SD(pricesF),4)
            diffM, diffF = round((meanM - meanF),4), round((meanF - meanM),4)
            data = {"desc": description, "M": [meanM, medianM, modesM, sdM, diffM], "F": [meanF, medianF, modesF, sdF, diffF]}
            res_dict[code]['data'] = data
            res_dict[code]['diff'] = diffF
    
        return res_dict
    
    def __code_description(self):
        res = dict()
        queries = session.query(MedCode)
        for query in queries:
            res[query.code] = query.description
        
        return res
    
    def __all_transactions(self):
        res = session.query(Transactions, Employee).filter(Transactions.emp_id == Employee.emp_id)
        return res
    
    def __transactions_gender(self, query, gender):
        res = query.filter(Employee.gender == gender)
        return res
    
    def __filter_by_med_code(self, query, code):
        res = query.filter(Transactions.medical_code == code)
        return res
    
    def __get_proc_price(self, query):
        res = [tr.procedure_price for tr, em in query]
        return res


class ProjectDescription:

    def project_description(file):
        texts = []
        with open(file, 'r') as file:
            line = file.readline()
            cnt = 0
            while line:
                if len(line) > 5:
                    texts.append(line)
                line = file.readline()
                cnt += 1
        return texts
                




