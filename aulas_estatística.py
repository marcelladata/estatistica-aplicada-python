"""## regressão linear"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo (variável independente X e dependente Y)
x = np.array([[1], [2], [3], [4], [5]])  # X como matriz (necessário para o scikit-learn)
y = np.array([1, 2, 3, 4, 5])  # valores de y

# Criando o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo aos dados
model.fit(x, y)

# Exibindo os coeficientes da regressão
print(f"Coeficiente (inclinação): {model.coef_[0]}")
print(f"Intercepto: {model.intercept_}")

# Fazendo uma previsão
x_new = np.array([[6]])  # valor novo para fazer a previsão
y_pred = model.predict(x_new)
print(f"Previsão para x = 6: {y_pred[0]}")

# Visualizando a regressão
plt.scatter(x, y, color='blue')  # gráfico de dispersão dos dados originais
plt.plot(x, model.predict(x), color='red')
plt.title('Regressão Linear Simples')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de exemplo com variáveis dispersas
np.random.seed(0)  # Definindo semente para reprodutibilidade
x = np.random.rand(100, 1) * 10  # Variável independente (dispersa entre 0 e 10)
y = 2 * x + np.random.randn(100, 1) * 2  # Variável dependente (com ruído)

# Criando o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo aos dados
model.fit(x, y)

# Exibindo os coeficientes da regressão
print(f"Coeficiente (inclinação): {model.coef_[0][0]}")
print(f"Intercepto: {model.intercept_[0]}")

# Fazendo uma previsão
x_new = np.array([[6]])
y_pred = model.predict(x_new)
print(f"Previsão para x=6: {y_pred[0][0]}")

# Visualizando a regressão
plt.scatter(x, y, color='blue')  # Gráfico de dispersão dos dados originais
plt.plot(x, model.predict(x), color='red')
plt.title('Regressão Linear com Variáveis Dispersas')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""## Correlação

"""

# Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Dados de exemplo: Temperatura (°C) e Vendas de Agasalhos (unidades)
dados = {
    'temperatura': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'vendas_agasalho': [500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
}

# Criando um DataFrame
df = pd.DataFrame(dados)

# Calculando a correlação
correlacao = df.corr()

print("Matriz de Correlação:")
print(correlacao)

# Visualizando a relação com um gráfico de dispersão
plt.scatter(df['temperatura'], df['vendas_agasalho'], color='blue')
plt.title('Correlação Negativa: Temperatura vs Vendas de Agasalhos')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Agasalhos (unidades)')
plt.show()

# Calculando a correlação de Pearson
correlacao_pearson, p_valor = pearsonr(df['temperatura'], df['vendas_agasalho'])
print(f"Correlação de Pearson: {correlacao_pearson}")
print(f"P-Valor: {p_valor}")

"""## Modelo com Machine Learning

"""

# Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Corrigido: model_selction para model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados de exemplo: Horas de estudo vs pontuação exame
horas_estudo = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
pontuacao = np.array([30, 35, 50, 60, 65, 70, 80, 85, 90, 95])

# Dividindo os dados em treino e teste (80% treino, 20% teste)
x_train, x_test, y_train, y_test = train_test_split(horas_estudo, pontuacao, test_size=0.2, random_state=42)  # Corrigido: ranom_state para random_state

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(x_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = modelo.predict(x_test)

# Avaliando o modelo com erro quadrático médio
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio: {mse}")

# Visualizando a linha de regressão
plt.scatter(horas_estudo, pontuacao, color='blue', label='Dados Reais')
plt.plot(horas_estudo, modelo.predict(horas_estudo), color='red', label='Linha de Regressão')
plt.xlabel('Horas de Estudo')
plt.ylabel('Pontuação')
plt.title('Regressão Linear: Horas de Estudo vs Pontuação')
plt.legend()
plt.show()

