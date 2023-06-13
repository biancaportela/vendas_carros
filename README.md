# Análise Economética e Preditiva de Venda de Carros Usados
- [Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- [Dicionário dos dados](https://github.com/biancaportela/vendas_carros/blob/main/dados/dicionario.md)
- [Limpeza dos dados](https://github.com/biancaportela/vendas_carros/blob/main/limpeza.ipynb)
- [Análise Econométrica: Projeto Final](https://github.com/biancaportela/vendas_carros/blob/main/analise_econometrica.ipynb)
- [Análise Preditiva: Projeto Final](https://github.com/biancaportela/vendas_carros/blob/main/analise_preditiva.ipynb)

# 1. Problema de negócios e objetivo da análise

O mercado de carros usados é uma indústria em constante crescimento. Os preços mais acessíveis dos veículos seminovos e usados frente aos modelos zero quilômetro contribuem para a expansão desse segmento, principalmente no Brasil, local que os preços dos carros novos estão muito [elevados](https://diariodocomercio.com.br/economia/montadoras-de-carro-projetam-2023-como-morno/);

O surgimento de portais online como o Craiglist tem facilitado boas informações aos clientes e vendedores sobre as tendências e padrões que determinam o valor do carro usado no mercado. A análise econométrica permite entender quais fatores afetam o preço de um carro e a utilização de machine learning pode ajudar a prever o valor de varejo de um carro, com base em um determinado conjunto de características.

Sendo assim, esse projeto foca em duas frentes: A análise inferencial e preditiva. O objetivo final é fazer um modelo preditivo que forneça modelos de previsão de preços ao público, a fim de ajudar as pessoas que desejam comprar ou vender carros e oferecer a elas uma melhor compreensão do setor automotivo. 


Para os clientes, comprar um carro usado de um revendedor pode ser uma experiência frustrante e insatisfatória, pois alguns revendedores são conhecidos por utilizar táticas de venda enganosas para fechar negócios. Portanto, para ajudar os consumidores a evitar cair em tais táticas, este estudo espera fornecer às pessoas as ferramentas certas para guiá-las em sua experiência de compra.

Já para os revendedores, um modelo de predição de carros usados permite prever os preços de revenda com base nas características dos veículos, ajudando a definir preços competitivos, tomar decisões informadas na compra de novos veículos, gerenciar eficientemente o estoque, negociar com clientes com base em estimativas precisas e obter insights valiosos sobre as tendências do mercado. Tendo esses conhecimentos é possível ao revendedor tomar decisões estratégicas, maximizar lucros e destacar-se em um mercado crescente e cada vez mais competivo.

# 2. Fases da análise

Como dito previamente, a análise tem duas fases principais: 

- **Análise econométrica**: busca-se entender como as características influenciam no preço final do carro, principalmente em relação à kilometragem. O objetivo final é responder a perguntas como: se eu aumentar a variável odomêtro em 1% quantos % o preço diminui/aumenta?

Os passos tomados na análise econométrica foram:

![Analise econometrica](https://github.com/biancaportela/vendas_carros/blob/main/imagens/Venda%20de%20carros.png?raw=true)

- **Análise preditiva**: Aqui constroe-se um modelo de previsão de preços. A ideia é treinar o modelo para que o cliente utilize características do carro para melhor auxilia-lo na compra ou venda de um carro a um preço competitivo.

Os passos tomados na análise preditiva foram:
![Analise preditiva](https://github.com/biancaportela/vendas_carros/blob/main/imagens/analise%20preditiva.png?raw=true)

# 3. Limpeza dos dados e análise exploratória

## 3.1 Dados ausentes

Eu iniciei esse projeto realizando alguns processos básicos de limpeza de dados tais como checar por duplicatas, valores ausentes e eliminando colunas que não seriam úteis na análise.

![ausentes](https://github.com/biancaportela/vendas_carros/blob/main/imagens/valores%20ausentes.png?raw=true)

O primeiro problema encontrado no dataset é a quantidade de dados ausentes. É preciso tratar cada coluna de maneira diferenciada para obter os melhores resultados. Vendo esses valores em números e em relação à amostra total temos que:
| Coluna       | Número de nulos | Porcentagem    |
|---------------|------------|------------|
| size          | 306,361    | 71.767%    |
| cylinders     | 177,678    | 41.622%    |
| condition     | 174,104    | 40.785%    |
| drive         | 130,567    | 30.586%    |
| paint_color   | 130,203    | 30.501%    |
| type          | 92,858     | 21.753%    |
| manufacturer  | 17,646     | 4.134%     |
| title_status  | 8,242      | 1.931%     |
| model         | 5,277      | 1.236%     |
| odometer      | 4,400      | 1.031%     |
| fuel          | 3,013      | 0.706%     |
| transmission  | 2,556      | 0.599%     |
| year          | 1,205      | 0.282%     |
 
 -  `size`: Com 71% dos dados da variável `size` sendo ausentes, não faz sentido manter ela no modelo, portanto esta coluna foi eliminada.
-  `condition`: Para lidar  com os valores faltantes na coluna `Condition` segui o exemplo de outros notebooks do Kaggle 
[[1](https://github.com/mo-adi/used_cars_pricing)] [[2](https://www.kaggle.com/code/msagmj/data-cleaning-eda-used-cars-prediction-86)]: imputei os valores faltante utilizando a coluna de `odometer`. Obti a média do `odometer` em cada `condition` e usei essas médias em relação a cada condição para imputar os valores. Entretanto, qualquer carro em que a label do ano fosse de 2022 ou mais novo recebeu a condição de novo, independente de quanto foi seu hodomêtro. 
- `title_status`: aos valores ausentes foi imputado a categoria `missing`.
- `cylinder`: com 40% dos dados ausentes, não foi possível apenas dropar os valores ausentes. A solução encontrada foi substituir os valores ausentes com a moda. Não é a solução ideal, pois pode gerar algum viés para o modelo, mas como não é possível falar com um time de negócios, esta foi a melhor saída encontrada.
- `type`: a variável type tem por volta de 20% dos dados ausentes. As categorias tem valors muito parecidos e se eu imputasse pela moda poderia gerar um viés desproporcional. Também não é possível dropar 20% dos dados. Por esses motivos, optei por colocar esses valores na categoria `other`.
- `drive`/`paint_color`: essas variáveis foram preenchidas utilizando a técninca de Forward Fill. Esta técnica é frequentemente utilizada quando os dados faltantes são considerados aleatórios. Esse método foi escolhido por que os valores das categorias nas duas variáveis são próximos e não faria sentido dropar os valores nulos ou criar uma nova categoria/imputar moda.
- As variáveis que possuiam menos de 5% de valores nulos, tiveram esses valores eliminados através do comando `dropna()`. Essas variáveis foram: `transmission`, `model`, `manufacturer`, `fuel`.

## 3.2 Outliers

O próximo desafio na limpeza de dados foi lidar com os outliers nas variáveis númericas. Os dados contém uma quantidade considerável de dados discrepantes, o que pode afetar a qualidade geral do modelo. A detecção de outliers foi feita com auxílio de análise gráfica, através de boxplots e levando em consideração outros trabalhos que utilizam esta base de dados.

### 3.2.1 Variável preço

Por exemplo, na variável preço temos que:
![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/boxplot_price.png?raw=true)

O número de outiliers é tão grande que não é possível ver a distribuição. Para lidar com esses dados, eu filtrei o dataset até que houvesse um intervalo razoável de valores. Assim, foram eliminados todos os veículos que tivessem um preço superior a $150.000 e os veículos que custassem menos de $500 doláres. A distribuição final ficou dessa maneira:

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/boxplot_price_clean.png?raw=true)

Podemos ter uma noção mais real da verdadeira distribuição. Ainda temos um número considerável de outliers. Como esses valores estão próximos uns dos outros, resta o questionamento se são *realmente* valores discrepantes e se sua retirada irá afetar o modelo.

Procedimento similar foi feito nas variáveis de ano e hodomêtro. Na variável de ano foram eliminados todos os valores que datavam pré 1960. Em `odometer` foram todos os valores acima de 400000 e iguais a 0. O boxplot final se dá por:

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/outlier_odo.png?raw=true) 

- Saimos de 426880 para 376778 observações após limpeza dos dados.

# 4. Análise exploratória de dados

O principal objetivo desta fase da análise é o entedimento das distribuições do dataset. Para tal, efetuei principalmente análises gráficas através de gráficos de barra, histogramas, gráficos de dispersão e a utilização de tabelas de discrição para obter medidas de tendência central e dispersão.

Alguns resultados interessentes que temos é que:

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/carros_distribuicao.png?raw=true) 
- A maioria dos carros tem condição excelente ou boa, com essas duas categorias somando mais de 60% do dataset.

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/preco_hist.png?raw=true) 

- Em relação ao preço, parte da distribuição se concentram entre preços até 40000 doláres, mas ainda assim é uma distribuição assimétrica.

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/precoxodometro.png?raw=true)

- Parece que, em nível, odômetro está negativamente relacionado com preço: quanto maior o número do odômetro, menor o preço.

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/anosxodometro.png?raw=true) 

- Carros entre 2000-2015 tem maior número milhas rodadas, enquanto carros mais velhos tem um registro de milhas menor. O senso comum sugere o contrário, que quanto mais velho o carro maior seja o valor do odômetro. Mas talvez isso seja apenas um viés no dataset: os carros mais velhos do que 2000 ainda estão sendo vendidos em carros usados justamente por serem mais conservados.

| condition   | fuel | odometer      | paint_color | price        | year   |
|-------------|------|---------------|-------------|--------------|--------|
| excellent   | gas  | 110859.908258 | white       | 15286.973199 | 2013.0 |
| fair        | gas  | 192387.356277 | white       | 8863.457109  | 2007.0 |
| good        | gas  | 76179.343139  | white       | 21312.202837 | 2018.0 |
| like new    | gas  | 88597.615859  | white       | 18648.255878 | 2015.0 |
| new         | gas  | 40040.062678  | white       | 29181.179606 | 2018.0 |
| salvage     | gas  | 152441.788409 | white       | 12372.417677 | 2008.0 |

Na tabela acima temos a média de odomêtro e preço de acordo com a condição do carro. Além disso, temos a moda do fuel, a cor e o ano. Assim, temos que a maioria dos carros salvage são de 2018 e a maioria dos carros considerados novos são de 2018. Aqui temos que:

- A quilometragem registrada no hodômetro varia para cada condição do carro. Carros em condição "fair" (justo) possuem a maior média de quilometragem, seguidos por carros em condição "salvage" (de salvamento). Carros em condição "new" (novo) possuem a menor média de quilometragem.

- O preço médio dos carros varia para cada condição. Carros em condição "good" (bom) possuem o preço médio mais alto, seguidos por carros em condição "excellent" (excelente). Carros em condição "fair" (justo) possuem o preço médio mais baixo.

# 5. Modelo econométrico

## 5.1 Preprocessamento dos dados

### 5.1.1 Linearizando variáveis

O principal objetivo da transformação das variáveis é tornar linear uma relação não linear, embora existam outros motivos, por exemplo, a obtenção de uma distribuição normal, aliviar problemas de heterocesticidade e obter estimativas menos sensíveis a outliers. O pressuposto da linearidade  é o alicerce do modelo de regressão clássica.
O uso de logarítimos é um dos principais instrumentos matemáticos utilizados na modelagem estatística. A utilização do logaritmo natural é preferida nesse caso, em razão de uma propriedade interessante do logaritmo natural: pequenas variações no logaritmo natural representam variações percentuais na variável em análise.
Dessa maneira, optou-se por fazer a transformação logaritmica do `price` e `odometer`.  Foi possível deixa-las mais próximas da normalidade, como podemos ver pela variável `price`:

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/ln%20preco.png?raw=true)

Ainda assim as variáveis ainda apresentem cauda longa, o que pode ser comprovado por um teste de normalidade. A relevância dos testes de normalidade reside no fato que algumas estratégia estatisticas são baseados no pressuposto de normalidade. Esse é o caso da regressão linear por mínimos quadrados ordinários.

É possível utilizar o Q-Q Plot para confirmar se as variáveis apresentam distribuição normal. O Q-Q plot é um gráfico que permite realizar a comparação visual dos quartis da amostra com os quartis teóricos para uma distribuição normal. Se as observações se distanciam da linha reta, a variável não apresenta distribuição normal. Esse não é o caso das variáveis analisadas:

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/normalidea_base_p.png?raw=true)

Como temos muitos dados e alguns outliers que fazem a distribuição ser assimétricas, assumi que os dados tendem a normalidade pelo teorema do limite central e prossegui com a regressão.
Após, testei a normalidade dos resíduos, que é pressuposto para o modelo linear clássico. No caso de falha, teremos que lidar com este problema de maneiras mais sofisticadas.

### 5.1.2 Correlação

O objetivo dessa primeira parte da análise é identificar alguma relação causal de interesse. A primeira pergunta que devemos nos fazer é se existe alguma associação entre as variáveis e, em caso positivo, qual a extensão dessa associação. Essa pergunta é respondida através da análise de correlação. 

O coeficiente de correlação permite calcular a direção e o grau de associação entre as variáveis. A covariância dos desvios padronizados é chamado de coeficiente de correlação de Pearson.Este coeficiente é a medida mais utilizada para a verificação preliminar da relação entre duas variáveis. 

O problema é que o coeficiente de correlação de Pearson é sensível à presença de outliers. Podemos utilizar o Spearman's Rho, que não requer que os dados apresentam distribuição normal.

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/correlacao.png?raw=true) 

- Podemos ver que `ln_odometer` e `ln_price` estão correlacionados de maneira negativa, de forma forte. Isso indica que quanto maior o valor do odomêtro menor o  preço.

- O preço e o ano está correlacionado positivamente, de forma forte. Isso significa que as duas variáveis se movem juntas, indicando que o preço tende a aumentar conforme o ano do carro for mais novo, o que faz sentido já que carros com anos mais altos devem ser mais novos.

- Existe uma correlação negativa forte entre ln_odometer e year. Isso indica que  em geral, carros mais recentes tendem a ter uma quilometragem menor registrada - novamente eles são provavelmente mais novos.

Iremos ver se estas correlações indicam alguma causalidade através dos modelos de regressão.

### 5.1.3 Tratando variáveis com alta cardinalidade

Antes de prosseguir com a regressão linear, melhorei o dataset para análise:

- Eliminei as variáveis  `region` e `price`, visto que elas possuem a mesma informação que outras colunas no dataset e poderiam gerar problemas de colineariedade dos dados.

- Diminui algumas categorias das variáveis categoricas, por questões de memória computacional, já que todas as variáveis categóricas virarão dummies.
    - Irei categorizar os estados por macro regiões.
    - Na variável `manufacturer` coloquei todas as categorias que correspondem a menos de 1% do do dataset em `others`.
- Dropei a variável `model` pois ela possui alta cardinalidade, o que pode levar a problemas de dimensionamento e complexidade computacional. Uma outra alternativa para lidar com elas em análise posterior seria realizar um PCA num pre-processamento dos dados.
- Fiz dummies para todas a variáveis categóricas. Uma das variáveis será eliminada, novamente com o objetivo de evitar colineariedade dos dados.

## 5.2 Modelo econométrico

A identificação de uma relação de causalidade entre duas variáveis de interesse pode ser representada através de uma equação, onde y é a variável dependente - o fenômeno que desejamos analisar - e x são as variáveis explicativas, que provoca variações em y. A análise dos betas é feita da seguinte maneira: tudo o mais constante, a variação na variável $x_j$ gera um impacto de $B_j$ na variável dependente.

Embora estejamos interessados em saber quanto a quilometragem afeta o preço do carro, outras variáveis também podem influenciar no preço final passado ao consumidor. Por isso as incluímos na regressão, de modo que possam servir como controle.

Em busca de analisar relações causais utilizei 4 modelos com diferentes formas funcionais. Sendo eles um log-log, log-nivel, log-nivel com termo quadrático e nivel-nivel com termo quadratico. O único que não apresentou problema de colineariedade nos dados foi o log-log, após a eliminação da coluna `year`. É ele que analiso aqui, embora seja possível econtrar detalhes dos outros modelos no notebook.

**obs**: em todos os modelos irei utilizar o MQO robusto para heterocedasticidade. A presença de heterocedasticidade é uma violação dos pressupostos do modelo linear clássico, mas sua presença é muito comum. Por isso preferi utilizar as regressões robustas.

A forma funcional utilizada foi:

$$ ln\_price = \beta_0 + \beta_1 ln\_odometer + \beta_{carro}caracteristicas\_carro + u$$

Em que $\beta_{carro}$ correspondem  a 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state'
Nosso modelo é um caso de modelo log-log, ou modelo de elasticidade constante, em que $\beta_1$ é a elasticidade de y em relação a x, ou seja, a variação percentual de y dado uma variação percentual em x. 


![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/ols%20results.png?raw=true)

-O coeficiente de `ln_odometer` é a elasticidade do preço em relação ao odomêtro. Assim, quando o odomêtro aumenta em 1% o preço cai em 0.1183%, tudo mais constante.

- O R-quadrado é uma medida da proporção da variabilidade na variável dependente (ln_price) que é explicada pelo modelo. Nesse caso, temos que cerca de 43% da variabilidade em ln_price é explicada pelas variáveis independentes incluídas no modelo.

## 5.3 Testando as hipótese do modelo de regressão linear

As hipóteses têm um papel importante na análise de relações de causalidade: como não temos todo o universo para trabalhar e nem sempre é possível obter dados a partir de experimentos controlados, utilizamos amostras aleatórias supostamente representativas do universo que estamos estudando. Desta forma, as hipóteses permitem que a interpretação dos resultados seja extrapolada de forma ampla e generalizada, e garantem que as propriedades dos estimadores de MQO serão mantidas. Logo, caso elas sejam respeitadas  teremos um modelo não viesado, eficiente e consistente. As principais premissas são:

1. **Hipótese 1**: As variáveis independentes e o termo de erro são não correlacionados -> caso essa hipótese seja violada, temos problemas de endogeneidade e os estimadores de MQO serão viesados e incosistentes.
2. **Hipótese 2**: Colineariedade não perfeita entre as variáveis independentes. A hipótese requer que nenhum dos regressores seja uma função linear perfeita dos outros regressores -> caso seja quebrada temos um problema de multicolineariedade e os estimadores do MQO são ineficientes.
3. **Hipótese 3**: Os termos de erro tem variância uniforme e não são correlacionados uns com os outros. Se a variável se altera com qualquer uma das variáveis explicativas, então temos heterocedasticidade -> Nesses casos os estimadores MQO são ineficientes, porém não viesados.
4. **Hipótese 4**: Os termos de erro são normalmente distribuídos. Quase nunca acontece, mas a Lei dos Grandes Números garante que, em uma grande amostra, o termo de erro apresenta distribuição que se aprocima da distribuição normal.

### 5.3.1 Problemas com o erro: Heterocedasticidade

O problema de heterocedasticidade acontece quando os termos de erro condicional das variáveis explicativas deixa de ser constante.A existência da heterocedasticidade não traz nenhuma implicação sobre a ausência de viés dos estimadores MQO e eles continuam não tendenciosos e consistentes. Mas as estatísticas t, F e LM deixam de ser válidas e temos que usar regressões robustas à heterocedasticidade.

O teste que usaremos aqui é o teste de Breusch-Pagan. A ideia básica do teste é verificar se termos de erro ao quadrado ($u^2$) está relacionado, em valor esperado, a uma ou mais variáveis independentes. A hipótese nula é:
$$ H_0: E(u^2|x_1, x_2, \ldots, x_k) = E(u^2) = \sigma^2$$

Se $H_0$ for falsa, o valor esperado de $u^2$, dadas as variáveis independentes, pode ser qualquer função de $x_j$

Como nunca conheceremos de fato os erros no modelo populacional, utilizamos as estimativas dos erros, os resíduos do MQO. Se o p-valor for suficientemente pequeno então rejeita-se a hipótese nula de homocedasticidade. Caso contrário, não podemos rejeitar a hipótese de homocedasticidade e temos erros heterocedásticos.

Aqui está a tabela com os resultados fornecidos:

| Resultado                  | Lagrange multiplier statistic | p-value | f-value | f p-value |
|----------------------------|-------------------------------|---------|---------|-----------|
| Resultado log-log          | 26021.682961435086           | 0.0     | 369.2894828618754  | 0.0       |
| Resultado log-nivel        | 13914.507312537402           | 0.0     | 190.48576155640865 | 0.0       |
| Resultado nivel-nivel      | 14562.215575099131           | 0.0     | 197.13612972774752 | 0.0       |
| Resultado log-log quadratico | 17082.582831214266           | 0.0     | 232.97401421521542 | 0.0       |

Por o p-valor ser baixo (0.0), temos evidências suficientes para rejeitar a hipótese nula e concluir que há heteroscedasticidade presente. Isso se repete em todos os modelos testados.

### 5.3.2: Testes para problema com os regressores: multicolineariedade

A hipótese do modelo de regressão linear clássico diz que a colineariade entre as variáveis independentes não pode existir de forma perfeita. A colineariedade refere-se à situação em que duas variáveis independentes são fortemente correlacionadas,  enquanto a multicolinearidade é a situação em que há mais de duas variáveis explicativas com alto grau de correlação.

Embora problemas de multicolineariedade não gerem modelos viesados, a presença dela faz com que o modelo deixe de apresentar a menor variância possível e, portanto, o estimador de MQO perde em eficiência e maior sua probabilidade  de obter estimativas pontuais erráticas.

Vamos checar por multicolineariedade com o Teste Fator de Inflação da Variância (VIF).  O teste do Fator de Inflação de Variância fornece medidas de impacto da colineariedade entre as variáveis explicativas e o modelo de regressão sobre a precisão da estimativa. O menor valor do VIF é 1 e um valor maior do que 10 é indicação de problema potencial de multicolineariedade.

Utilizaremos uma função para calcular o VIF, retirada de um artigo do [Towards Data Science](https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b)

Aqui está a tabela com a variável e seus respectivos valores VIF:

| Variável                    | VIF        |
|-----------------------------|------------|
| cylinders_6 cylinders       | 80.901659 |
| cylinders_4 cylinders       | 53.887224 |
| cylinders_8 cylinders       | 49.156709 |
| manufacturer_ford           | 10.655899 |
| manufacturer_chevrolet      | 8.762418  |
| ...                         | ...        |
| title_status_rebuilt        | 1.017283  |
| paint_color_purple          | 1.009400  |
| title_status_salvage        | 1.007168  |
| title_status_lien           | 1.003593  |
| title_status_parts only     | 1.000898  |


As variáveis que possuem alta colineariedade são as dummies de cylinders e a `manufacturer_ford`. A dos cilindros é muito maior do que o máximo que estamos permitindo (10). Isso indica que ela deveria ser tratada de uma maneira especial. Após a retirada do modelo, o problema com colieariedade na dummy de cilindros é resolvido.

Poderiamos lidar com essas dummies com alta colineariedade através de uma proxy ou combina-las em único índice. Isso pode ser feito em um trabalho posterior.

### 5.3.3:  Problemas com os regressores: testes de especificação e endogeneidade

Estatísticamente, podemos realizar alguns tipos de teste que permitem identificar se o modelo escolhido apresenta ou não algum problema de erro de especificação de forma funcional. Neste trabalho usaremos o teste de erro de especificação da regressão de Ramsey (RESET), que também se mostra bastante poderoso para detecção de não lineariedade.

A intuição do teste é de que, se existem erros de especificação, a hipótese média condicional do termo de erro deixa de ser satisfeita. Se isso ocorre, nenhuma função não linear das variáveis idependentes deve ser significante quando adicionada ao modelo original. Portanto, para realizar o teste RESET temos que decidir quantas funções dos valores estimados devem ser incluídas na regressão. 

A hipótese nula é que nossa equação original está corretamente especificada. Portanto, a estatística do teste RESET é a estatística F para testar $H_0: \sigma_1 = 0, \sigma_2 = 0$ no nosso modelo expandido com as novas funções dos valores estimados. Uma estatística F significante sugere algum tipo de problema na forma funcional.

Aqui está a tabela com os resultados do teste Ramsey-RESET para a forma funcional correta:

| Modelo                     | Variável                        | Estatística F | Valor p |
|----------------------------|---------------------------------|---------------|---------|
| log-log                    | Ramsey-RESET Test F-Statistic   | 8102.914777   | 0.0     |
| log-log                    | Ramsey-RESET Test P-value       | -             | 0.0     |
| log-nivel                  | Ramsey-RESET Test F-Statistic   | 1114.296285   | 0.0     |
| log-nivel                  | Ramsey-RESET Test P-value       | -             | 0.0     |
| log-nivel quadratico       | Ramsey-RESET Test F-Statistic   | 4430.428697   | 0.0     |
| log-nivel quadratico       | Ramsey-RESET Test P-value       | -             | 0.0     |
| log-log quadratico         | Ramsey-RESET Test F-Statistic   | 8987.905469   | 0.0     |
| log-log quadratico         | Ramsey-RESET Test P-value       | -             | 0.0     |


Todos nossos modelos rejeitam a hipótese nula, mesmo utilizando paramêtros de robustez. Para todos os modelos testados, os valores elevados das estatísticas F e os p-valores nulos indicam que há evidências significativas para rejeitar a hipótese nula de que os modelos estão corretamente especificados. 

Isso significa que temos uma má especificação no modelo. A má especificação do modelo pode acontecer se a relação entre a variável dependente e as variáveis independentes não forem lineares, as variáveis se relacionarem de forma iterativa e não de forma aditiva ou ainda de não termos considerados alguma variável relevante (problema de variável omitida). 

No caso de variável omitida, temos que os nossos dados não são suficiente para entender as relações de causalidade no modelo. A solução seria arranjar mais dados, ou utilizar alguma variável de proxy para eliminar ou reduzir este problema. Outra forma para resolver o problema de endogeneidade de uma ou mais variáveis explicativas é pelo uso de variáveis intrumentais. Todas essas opções iriam requerer um estudo mais aprofundando sobre o mercado de carros usados e da literatura de preços e demanda, talvez até consulta com especialistas.


### 5.3.4: Teste de normalidade dos resíduos

![Alt text](https://github.com/biancaportela/vendas_carros/blob/main/imagens/residuos%202.png?raw=true)

Quando rodamos o teste de normalidade nos resíduos, vemos que eles se distanciam da linha vermelha. Mais uma violação do MQO.


## 5.4 Análise econométrica: conclusões e próximos passos

O que podemos concluir com todo esse estudo é que os dados da maneira que temos aqui são insuficentes para conseguir encontrar alguma relação de causalidade entre preço e odomêtro. Talvez sejam necessários mais dados ou talvez devemos realizar alguma transformação nas variáveis, como um PCA, para tentar diminuir os problemas com colineariedade. Podemos também utilizar o MQG, que é robusto a ouliers. Outra forma é utilizar regressões não paramétricas, para tentar melhorar o ajuste do modelo.

Outro passo que pode ser feito é realizar testes de estatística de influência.  Na presença de outliers, o MQO não reflete com precisão a relação estatística entre as variáveis, sendo um dos motivos para tal o fato de que outliers fazem com que a distribuição não seja normal. Os testes de influência permitem identificar essas observações e verificar como os resultados obtidos na regressão são sensíveis à sua presença. Esse dataset ainda contém diversos outliers, que podem estar prejudicando a performance do modelo. A aplicação desses testes e o tratamento de outliers podem gerar insights mais significativos.


# 6. Análise preditiva

Nessa parte do projeto buscamos fazer uma análise preditiva do preço dos carros seminovos a serem vendidos, baseados em suas features.

**Objetivo:** Prever o preço dos carros baseados em suas features.