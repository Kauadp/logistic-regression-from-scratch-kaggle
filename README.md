# 📊 Regressão Logística do Zero — Predição no Kaggle (Sem Bibliotecas de ML)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange?logo=numpy)
![No ML Libraries](https://img.shields.io/badge/Machine%20Learning%20Libraries-None-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Score%200.74401-20beff?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-success)

--- 


## 📌 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning do zero, sem uso de bibliotecas de ML como **scikit-learn**, para resolver um problema de classificação binária do Kaggle utilizando Regressão Logística otimizada por Gradiente Ascendente.

O objetivo principal do projeto é consolidar o entendimento **matemático** dos modelos de classificação, indo além do uso de bibliotecas prontas e implementando cada etapa do pipeline manualmente
 - Pré-processamento de dados
 - Padronização de variáveis
 - Análise de Componentes Principais (PCA)
 - Estimação por Máxima Verossimilhança (MLE)
 - Otimização por métodos de gradiente

Score final no Kaggle (public leaderboard): **0.74401**

---

## 🧠 Conceitos Implementados

O projeto cobre tanto teoria estatística quanto otimização numérica:

 - Análise Exploratória de Dados (EDA)
 - Tratamento de valores ausentes (missing values)
 - Padronização (z-score)
 - PCA implementado manualmente via matriz de correlação e autovalores
 - Regressão Logística implementada do zero
 - Máxima Verossimilhança para dados Bernoulli
 - Gradiente Ascendente para otimização
 - Geração de predições e arquivo de submissão para o Kaggle

Nenhuma biblioteca de Machine Learning foi utilizada para o treinamento do modelo.

---

## 🧮 Modelos Matemáticos Utilizados

### 📈 Regressão Linear — Imputação da Variável *Age*

Para imputar valores ausentes da variável **Age**, foi utilizado um modelo de **Regressão Linear**, assumindo uma relação aproximadamente linear entre a idade e outras variáveis explicativas do conjunto de dados.

Dado:

- Matriz de variáveis explicativas: X ∈ ℝⁿˣᵖ  
- Vetor resposta: y ∈ ℝⁿ (idade)

Modelo:

y = Xβ + ε  
onde: ε ~ N(0, σ² I)

O estimador de Mínimos Quadrados Ordinários (MQO) é dado por:

β̂ = (Xᵀ X)⁻¹ Xᵀ y

As idades ausentes são imputadas por:

ŷ = X β̂

Dessa forma, a imputação preserva relações estatísticas entre as variáveis, em vez de utilizar valores constantes como média ou mediana.

---

### 📊 Regressão Logística — Classificação Binária

O modelo de Regressão Logística foi utilizado para resolver o problema de **classificação binária** do Kaggle, estimando a probabilidade de ocorrência da classe positiva.

Dado:

- Matriz de variáveis: X ∈ ℝⁿˣᵖ  
- Vetor de parâmetros: β ∈ ℝᵖ

Preditor linear:

z = Xβ

Função sigmoide:

p = σ(z) = 1 / (1 + e^(−z))

onde pᵢ = P(Yᵢ = 1 | Xᵢ)

Assumindo:

Yᵢ ~ Bernoulli(pᵢ)

A log-verossimilhança do modelo é:

ℓ(β) = yᵀ log(p) + (1 − y)ᵀ log(1 − p)

O gradiente da log-verossimilhança é:

∇ℓ(β) = Xᵀ (y − p)

A estimação dos parâmetros é feita por **Gradiente Ascendente**:

β^(t+1) = β^(t) + α Xᵀ (y − p)

onde α é a taxa de aprendizado (*learning rate*).

Esse procedimento é iterado até convergência, maximizando a log-verossimilhança do modelo.


---

## ⚙️ Tecnologias Utilizadas
 
 - Python
 - NumPy
 - Pandas
 - Matplotlib (para visualizações básicas)

Sem uso de bibliotecas de Machine Learning para treinamento.

---

## 🎯 Resultados

 - Score público no Kaggle: 0.74401
 - Modelo: Regressão Logística otimizada por Gradiente Ascendente (maximização da log-verossimilhança)
 - Redução de dimensionalidade: PCA implementado manualmente


## 👤 Autor

**Kauã Dias**  
Estudante de Estatística e entusiasta de Ciência de Dados

- 🐙 GitHub: [github.com/Kauadp](https://github.com/Kauadp)  
- 🔗 LinkedIn: [linkedin.com/in/kauad](https://www.linkedin.com/in/kauad/)



## 📚 Referências Teóricas

Os conceitos estatísticos, probabilísticos e algébricos utilizados neste projeto foram estudados a partir das seguintes referências:

[1] James, G.; Witten, D.; Hastie, T.; Tibshirani, R.  
**An Introduction to Statistical Learning (ISLR)** — Springer.  
https://www.statlearning.com/  
→ Regressão linear, regressão logística, viés–variância, classificação.

[2] Stewart, J.  
**Cálculo — Volumes 1 e 2** — Cengage Learning.  
https://www.cengage.com.br/livro/calculo-volume-1/  
→ Derivadas, gradiente, otimização e fundamentos para métodos iterativos.

[3] Lay, D. C.; Lay, S. R.; McDonald, J. J.  
**Álgebra Linear e Suas Aplicações** — Pearson.  
https://www.pearson.com/en-us/subject-catalog/p/linear-algebra-and-its-applications/P200000006472  
→ Matrizes, autovalores, autovetores, base para PCA e regressão matricial.

[4] Boulos, P.  
**Geometria Analítica: Um Tratamento Vetorial** — McGraw-Hill.  
https://www.grupogen.com.br/livro/geometria-analitica-um-tratamento-vetorial-paulo-boulos  
→ Interpretação geométrica de vetores, projeções e espaços lineares.

[5] Cochran, W. G.  
**Sampling Techniques (Elementos de Amostragem)** — Wiley.  
https://www.wiley.com/en-us/Sampling+Techniques%2C+3rd+Edition-p-9780471162407  
→ Fundamentos de amostragem, inferência e variabilidade dos estimadores.

[6] Ross, S. M.  
**A First Course in Probability (Probabilidade)** — Pearson.  
https://www.pearson.com/en-us/subject-catalog/p/a-first-course-in-probability/P200000005405  
→ Variáveis aleatórias, distribuições, Bernoulli e modelos probabilísticos.

[7] Morettin, P. A.; Bussab, W. O.  
**Estatística Básica** — Saraiva.  
https://www.grupogen.com.br/livro/estatistica-basica-pedro-morettin-wilton-bussab  
→ Estatística descritiva, inferência, estimação e testes de hipóteses.