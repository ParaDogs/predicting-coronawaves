import os
import json
from datetime import datetime

covid_reports_files = './data/datasets/'
countries_info_file = './data/countries.json'

human_unit = 1000000
earth_population = 8000
zero_human = 1/(earth_population*2)

class Country_info_collector:
    def __init__(self):
        with open(countries_info_file, encoding='utf-8') as f:
            self.countries = json.load(f)
            f.close()

    def population(self, country):
        alt_spelling = self.alt_spelling(country)
        data = [r['population'] for r in self.countries if country in alt_spelling]
        if data: return data[0]
        return data/human_unit
    
    def gini(self, country):
        alt_spelling = self.alt_spelling(country)
        data = [r['gini'] for r in self.countries if r['name'] in alt_spelling]
        if data:
            s = str(data[0])
            if s == 'None': return 0
            return float(s)
        return 0

    def alt_spelling(self, country):
        return [r['altSpellings'] + [r['name']] + [r['alpha2Code']] + [r['alpha3Code']] for r in self.countries if country in r['altSpellings'] + [r['name']] + [r['alpha2Code']] + [r['alpha3Code']]][0]

    def borders(self, country, time):
        alt_spelling = self.alt_spelling(country)
        print(country)
        data = [r['borders'] for r in self.countries if r['name'] in alt_spelling][0]
        res = 0
        for n in data:
            res += self.infected(n, time) / self.population(n)
        if len(data) == 0: return 0
        res /= len(data)
        return res
    
    def infected(self, country, time):
        res = 0
        alt_spelling = self.alt_spelling(country)
        file_name = covid_reports_files + time + '.csv'
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(',')
                if (data[1] in alt_spelling):
                    if not data[3]: res += zero_human
                    else: res += int(data[3])
                elif (data[3] in alt_spelling):
                    if not data[7]: res += zero_human
                    else: res += int(data[7])
            f.close()
        return res/human_unit

    def dead(self, country, time):
        res = 0
        alt_spelling = self.alt_spelling(country)
        file_name = covid_reports_files + time + '.csv'
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(',')
                if (data[1] in alt_spelling):
                    if not data[4]: res += zero_human
                    else: res += int(data[4])
                elif (data[3] in alt_spelling):
                    if not data[8]: res += zero_human
                    else: res += int(data[8])
        f.close()
        return res/human_unit

    def recovered(self, country, time):
        res = 0
        alt_spelling = self.alt_spelling(country)
        file_name = covid_reports_files + time + '.csv'
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(',')
                if (data[1] in alt_spelling):
                    if not data[5]: res += zero_human
                    else: res += int(data[5])
                elif (data[3] in alt_spelling):
                    if not data[9]: res += zero_human
                    else: res += int(data[9])
        f.close()
        return res/human_unit

    def latlng(self, country):
        alt_spelling = self.alt_spelling(country)
        data = [r['latlng'] for r in self.countries if country in alt_spelling]
        if data: return data[0]
        return data
