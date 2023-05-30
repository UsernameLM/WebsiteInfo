import os
from selenium import webdriver
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def check_status(): #Ver se está conectando com o site
    my_url = "https://finance.yahoo.com/news"
    response = requests.get(my_url)
    # Pega exceção e formata(conecta as strings)
    print("response.ok : {} , response.status_code : {}".format(response.ok, response.status_code))
    print("Preview of response.text : ", response.text[:500])

def get_url(url):#Yahoo finance deixa usar o header default "User-Agent", mas talvez precise mudar dependendo do site
    response = requests.get(url)
    if not response.ok: #checa se a resposta da url retorna não válida válida(fora dos códigos 200)
        print('Código de status:', response.status_code)
        raise Exception('Falha em carregar url: {}'.format(url))
    page_content = response.content
    soup = BeautifulSoup(page_content, 'html.parser')
    return soup

def show_daily():
    page = get_url("https://www.infomoney.com.br/cotacoes/b3/indice/ibovespa/")
    divs = page.find_all("table")
    #Vai do fechamento até Volume print (divs[0].prettify)
    val_array = []  # create an empty array to store td tags
    cat_array = []
    for i, td in enumerate(divs[0].find_all('td')):  # i para o index e td para o valor do index
        if i % 2 == 0:  # pegar apenas os valores pares pois os impares são A Categoria
            cat_array.append(td.text)
        else:
            val_array.append(td.text)
    print(val_array)  # print the array of even-indexed td tags

def show_history():
    page = get_url("https://www.infomoney.com.br/cotacoes/b3/indice/ibovespa/historico/")
    table = page.find('table', id='quotes_history')
    tbody = table.find('tbody')
    rows = tbody.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if cells:
            cell_data = [cell.get_text(strip=True) for cell in cells]
            print(cell_data)
    print(rows)
    print(tbody)
    print(table)

def show_h():
    driver = webdriver.Chrome()
    # navegar pagina
    driver.get("https://www.infomoney.com.br/cotacoes/b3/indice/ibovespa/historico/")
    time.sleep(5)
    values = []
    values_usable = []
    # espera carregar a table
    table = driver.find_element(By.ID, "quotes_history")
    rows = table.find_elements(By.TAG_NAME, "tr")
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if cells:
            # extrar o primeiro e quarto valor e colocar eles em lista
            value_1 = cells[0].text.strip()
            value_2 = cells[1].text.strip()
            value_3 = cells[2].text.strip()
            value_4 = cells[3].text.strip()
            values.append([value_1, value_4])
            values_usable.append([value_2, value_3, value_4])
    # fecha driver e printa
    print(values_usable)
    return values_usable #temporario
    driver.quit()
    # pegar valor e converter em float e datas
    dates = [dt.datetime.strptime(value[0], '%d/%m/%Y') for value in values[1:]]
    value_2 = [float(value[1].replace(".", "").replace(",", ".")) for value in values[1:]]
    plt.plot(dates, value_2)
    plt.title('Ibovespa Historical Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m'))
    # prevent overlapping
    plt.xticks(rotation=45)
    # mostrar
    plt.show()

def predict(table):
    input_array = np.array([float(x.replace(',', '.')) for x in table])
    print(input_array)
    # Create dataframe from array
    df = pd.DataFrame(input_array)
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(input_array, test_size=0.2)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    # Define the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean squared error: ', mse)

    # Make predictions on new data
    predictions = model.predict(X)
    print('Predictions: ', predictions)



def predict2(table):
    arr = np.array(table)
    for i in range(1, len(arr)):
        if arr[i] == 'n/d':
            arr[i] = arr[i - 1]


    input_array = np.array([float(str(x).replace(',', '.')) for x in arr])
    print(input_array)
    # Create dataframe from array
    df = pd.DataFrame(table)
    print(df)
    x = df.iloc[:, :2]
    y = df.iloc[:, 2]
    print('Shape of X:', x.shape)
    print('Shape of y:', y.shape)
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(table, test_size=0.2)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)



predict2(show_h())