#!/usr/bin/env python
# coding: utf-8

# ![kbcra.png](attachment:kbcra.png)

# <table text-align="left"; style="width: 100%;"  >
# <tbody>
# 
# <tr text-align="center">
# <td width="19%" bgcolor="orange" ><FONT FONT SIZE="+1" COLOR="WHITE">Autores</FONT>&nbsp;</td>
# <td width="73%" bgcolor="WHITE" align="center"><FONT FONT SIZE="+1" COLOR="ORANGE" >Correo</FONT>&nbsp;</td>
# </tr>
# 
# <tr>
# <td>&nbsp;</td>
# <td> &nbsp;</td>
# <td> &nbsp;</td>
# <td>&nbsp;</td>
# </tr>
# <tr>
# <td width="5%"; bgcolor="orange" border = "4"><FONT FONT SIZE="+0.2" COLOR="WHITE">Valentina Díaz Torres</FONT>&nbsp;</td>
# <td width="5%"; bgcolor="WHITE"><FONT FONT SIZE="+0.5" COLOR="ORANGE">Valentina.diaz@cunef.edu</FONT>&nbsp;</td>
# </tr>
# 
# <tr>
# <td width="5%"; bgcolor="orange"><FONT FONT SIZE="+0.2" COLOR="WHITE">Francisco del Val Yague</FONT>&nbsp;</td>
# 
# <td width="5%"; bgcolor="WHITE"><FONT FONT SIZE="+0.5" COLOR="ORANGE">Francisco.delval@cunef.edu</FONT>&nbsp;</td>
# </tr>
# <tr>
# <td width="5%"; bgcolor="orange"><FONT FONT SIZE="+0.5" COLOR="WHITE">Alejandro García Girón</FONT>&nbsp;</td>
# 
# <td width="5%"; bgcolor="WHITE"><FONT FONT SIZE="+0.5" COLOR="ORANGE">A.garciagiron@cunef.edu</FONT>&nbsp;</td>
# </tr>
#     
# </tbody>
# </table>

# **Carga de Librerias**

# In[1]:


#librerías

import sys
ruta = sys.executable
import pandas as pd 
import numpy as np
import seaborn as sns
get_ipython().system("{sys.executable} -m pip install pandas-profiling 2>(grep -v 'Requirement already satisfied' 1>&2)")
from sklearn.preprocessing import OneHotEncoder
from pandas_profiling import ProfileReport # eda 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system("pip install sweetviz 2>(grep -v 'Requirement already satisfied' 1>&2) #Redirigimos la salida de errores, filtrando con el comando grep el inicio de los mensajes que aparecen.")
import sweetviz as sv
get_ipython().system("pip install pyod 2>(grep -v 'Requirement already satisfied' 1>&2)")
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
pd.options.display.max_columns = None #para poder visualizar todas las columnas sin puntos suspensivos
pd.options.display.max_rows = None #para poder visualizar todas las filas sin puntos suspensivos


# __Carga de datos__

# In[2]:


# carga y visualizacion de los datos
a = pd.read_csv('../../data/01_raw/Loan_training_set_1_4.csv', header=1,skipfooter=2,engine='python')
b = pd.read_csv('../../data/01_raw/Loan_training_set_2_4.csv', header=1,skipfooter=2,engine='python')
c = pd.read_csv('../../data/01_raw/Loan_training_set_3_4.csv', header=1,skipfooter=2,engine='python')
d = pd.read_csv('../../data/01_raw/Loan_training_set_4_4.csv', header=1,skipfooter=2,engine='python')


# In[3]:


#concatenamos los diferentes csv, utilizando axis=0 para que queden unidas en dirección descendente, ya que es necesario unir tres archivos csv

data = pd.concat([a,b,c,d], axis=0)


# #### LIMPIEZA DE VARIABLES PREVIAS

# Con un previo conocimiento de negocio consideramos a eliminar las variables que tenemos a continuación ya que representan datos que no aportan calidad a la predicción. El 'id' por hacer una aproximación al caso más cercano no nos representa más que información nominal más que de la persona a la que se le concede el prestramo. Hay variables que nos aportan información pero es despues de el momento que queremos predecir, la concesióin o no del prestamo, por las que también las obviamos.

# In[4]:


data = data.drop(["id", "grade","zip_code","addr_state", "addr_state","sub_grade","url", "issue_d", "policy_code",
                  "last_pymnt_d","next_pymnt_d","last_credit_pull_d","emp_title", "purpose","title",
                  "verification_status","earliest_cr_line","revol_util"], axis=1)


# Hemos eliminado algunas variables a priori, como `grade` y `subgrade`, `ID`, `url`, fechas y direcciones, ya que consideramos que no son significativas para el análisis y que vamos a quitarlas sin necesidad de hacer un EDA previo. En principio, a una persona  no se le concedería o no un crédito según el país en el que viva o su dirección, según hemos tenido en cuenta.

# ## EDA

# Nuestro EDA parte de 134 variables a analizar, en las que se explorarán valores nulos, duplicados, distribuciones y se visualizarán, para poder decidir con qué variables nos quedamos finalmente para implementar nuestros modelos.

# In[5]:


data.shape  #dimensión de nuestra base de datos con todas las observaciones 


# En el documento que se genera a continuación, 'eda.html', se crea un análisis exploratorio de todas las variables. Esto nos permite conocer el número de duplicados, que en este caso aparece que es 0 y el número de valores nulos. En cuanto a estos, es especialmente útil, ya que se muestra el porcentaje de valores nulos de cada variable, cambiando de color según el rango. Esto nos ha permitido conocer qué variables teníamos que tener en cuenta y cuáles no nos iban a aportar mucha información.

# In[6]:


#eda = sv.analyze(mini_data)  #esto genera el documento, que luego es guardado


# In[7]:


#eda.show_html('eda.html')   #con esto se pasa a html el documento generado


# __Comprobación de duplicados__

# In[8]:


# duplicados 
data.duplicated().sum()


# Como ya comprobamos en el documento 'eda.html', comentado anteriormente, corroboramos que no existe ningún valor duplicado detectado.

# __Histogramas y gráficos de dispersión__

# En primer lugar, generamos los histogramas de todas las variables, para comprobar su distribución y que esto nos ayude posteriormente la decisión de tener en cuenta o no según qué variables.
# 
# Sospechamos que algunas variables, según su representación, continen muchos valores nulos, o contienen una información poco útil para el análisis.

# In[9]:


data.hist(figsize = (25, 30), bins = 30, xlabelsize = 10, ylabelsize =8);


# ## Ingeniería de variables

# __Tratamiento de NA's__

# Tras comprobar que hay bastantes columnas que contienen la mayoría de datos nulos, hemos decidido estipular el 90% como la frontera para no aceptar trabajar con una variable. 
# 
# En primer lugar, estas columnas son detectadas por la función siguiente. Una vez que se estiman son eliminadas y nos quedaríamos por tanto con las que tengan un número inferior de na's. Hemos decidido elegir 90% porque si se elige un porcentaje inferior se estaría perdiendo mucha información. Queremos automatizar el proceso todo lo posible, por lo tanto no podemos evaluar columna por columna y es por eso que se crea esta función, para que se introduzcan los datos que se introduzcan en el modelos no cause problemas tales como el Overfitting. 

# In[10]:


def limpio(dataset): #Nombre de la función, que recibirá por parámetro un dataset
    etiqueta = dataset.columns #Almaceno en etiqueta los nombres de todas las columnas de mi dataset para realizar el bucle
    print(etiqueta)
    for i in range(len(etiqueta)): #Para la longitud de las columnas
        porcentaje = (dataset[etiqueta[i]].isnull().sum()/len(dataset))*100  #Almaceno en porcentaje de manera individual el porcentaje de valores NaN que tiene cada columna
        if(porcentaje > 90.0).any(): #Filtro, si el porcentaje de valores NaN es mayor que 90
            dataset = dataset.drop([etiqueta[i]],axis = 1) #Elimino esa columna de mi dataset
    return dataset #Devuelvo el dataset modificado


# In[11]:


data = limpio(data) #aplicamos la función "limpio" al dataset, para que elimine todas aquellas columnas que tienen más del 90% de valores na's


# Tras aplicar la función, comprobamos que las variables que nos quedan serían 96.

# In[12]:


data.shape #Comprobación columnas eliminadas


# ### Sustituir los valores NA's restantes, de las variables que tienen menos de un 90%.

# Con el resto de valores na's que nos han quedado, ha sido necesario tomar una decisión. Lo ideal para obtener un resultado más preciso sería evaluar variable por variable y así poder reemplazar estos valores por lo que fuese más adecuado para el caso. No obstante, consideramos que no es algo muy eficiente, de cara a estar preparados a futuros datos no contemplados. Es por eso, que finalmente se ha decidido sustituir los valores numéricos por la media y las columnas con valores categóricos por la mediana. Así, si aparece un nuevo valor na's será filtrado y tratado de forma automática.

# In[13]:


#dividir el dataset en variables categóricas y numéricas
numericas = data.select_dtypes(include = [np.number]) #Filtro por las que sean del tipo np.number
categoricas = data.select_dtypes(include = [np.object]) #Filtro por las que sean del tipo np.object
#sustituir las variables numéricas por la media
numericas = numericas.apply(lambda x: x.replace("", np.nan))
numericas = numericas.apply(lambda x: x.fillna(x.mean()))
#sustituir las variables categóricas por la mediana.
categoricas = categoricas.apply(lambda x: x.replace("", np.nan))
categoricas = categoricas.apply(lambda x: x.fillna(x.value_counts().index[0]))
#Unión de todas las variables
data = pd.concat([numericas, categoricas], axis = 1)


# A continuación hemos creado una función que detecte qué columnas son iguales, ya que analizando la base de datos, se ha detectado, que los valores de algunas columnas aparecen duplicados en otras, pero con nombres distintos, por lo que consideramos que estarían recogiendo la misma información

# In[14]:


#Función para comparar si dos columnas son iguales, y en caso de serlo, eliminamos la segunda bajo nuestro criterio
def columnasIguales(columna1, columna2): 
    for i in range(len(columna1)):
        if(data[columna1][i] == data[columna2][i]).all(): #Si todos los valores de columna1 y columna2 son iguales
            val = 1 #Devuelvo un 1 
        else: val = 0 #Si no, devuelvo un 0
    if(val == 1):
        print("Las columna son iguales") 
        dataset = data.drop([columna2],axis=1) #Si son iguales, elimino la segunda columna de mi dataset
    elif (val == 0):
        print("Las columnas son distintas") #Si no, no hago nada
    else: print("Error")
    return dataset #Devuelvo el dataset finalente 
    


# La función anterior detecta aquellas columnas que son iguales. Luego, la función se le aplica a estas detectadas de dos en dos , quedandonos solo con las primeras que aparecen en el paréntesis y desapareciendo las segundas.
# 
# Algunas columnas como loan_amnt y funded_amnt tienen los mismos valores, por lo que son detectadas por la función anterior y se elige solo una de ellas.

# In[15]:



#Comparación de columnas que a priori parecen iguales, pero necesitamos de la 
#función definida anteriormente para confirmarlo.

data = columnasIguales('loan_amnt','funded_amnt')
data = columnasIguales('loan_amnt','funded_amnt_inv')
data = columnasIguales('out_prncp','out_prncp_inv')
data['total_pymnt'] = round(data.total_pymnt,2) #Redondeamos para poder ejecutar la comparación
data = columnasIguales('total_pymnt','total_pymnt_inv')
data = columnasIguales('total_rec_late_fee','recoveries')
data = columnasIguales('total_rec_late_fee','collection_recovery_fee')
data = columnasIguales('acc_now_delinq','tot_coll_amt')


# Como la variable "term" contiene la palabra month seguida del número, se elimina el término "months", para poder quedarnos solo con el dato numérico que nos interesa.

# In[16]:


termNuevo = np.asarray(data.term) #Almacenamos en nuestra variable, un array de nuestra columna term para poder acceder bien a las posiciones del vector
arrayTerm = [] #Declaro un array vacío
for i in range(len(data.term)):
    arrayTerm.append(termNuevo[i].rstrip(' months')) #Recorro todas las filas eliminando de la derecha la cadena (' months') para quedarme únicamente con el valor del número
data['term'] = arrayTerm #Almaceno en la columna del dataset, el nuevo array limpio de elementos del tipo cadena


# En cuanto a la columna "int_rate" ofrece el valor en porcentaje, como el símbolo % no es válido para trabajar con el, este es eliminado, y el dato que queda aparece en porcentaje, pero sin el símbolo.

# In[17]:


#Exactamente igual que la función anterior, pero suprimiendo esta vez el símbolo de porcentaje (%)
rateNuevo = np.asarray(data.int_rate)
arrayRate = []
for i in range(len(data.int_rate)):
    arrayRate.append(rateNuevo[i].rstrip('%'))
data['int_rate'] = arrayRate


# La columna "emp_length" es complicada de trabjar. Los datos que contienen están en años, con la palabra "years" incluida, además de un símbolo "+" o "<" para indicar el rango en el que se encuentra. Por eso, ha sido necesario crear una función para la limpieza de esta variable, de tal forma que nos quedásemos solo con el número de años.

# In[18]:


letra = data.emp_length[0:len(data.emp_length)].astype(str) #Almacenamos en letra nuestra columna 'emp_length' convirtiéndola en tipo cadena
letra = np.asarray(letra) #La pasamos a array para poder recorrer las posiciones del vector
le = [] #Array vacío
for i in range(len(data.emp_length)):
    if(letra[i] == 'nan'): #Si el valor es NaN, lo sustituimos por ' years', que más tarde será un espacio en blanco tras la limpieza que haremos abajo
        letra[i] = ' years'
    elif(letra[i] == '< 1 year'): #Si el valor es '< 1 year', lo sustituiremos por el valor 0, ya que la columna está expresada en años y <1 es que no ha llegado todavía a constar como un año.
        letra[i] == '0 years'
    le.append(letra[i].lstrip('<').rstrip('+ years')) #Limpio tanto si tiene símbolo '<' como '+', además de lo restante 'years' 
data['emp_length'] = le #Almaceno en la columna del dataset el nuevo valor del array


# Como algunas variables son categóricas y no podemos trabajar con ellas en los modelos posteriores, es necesario pasarlas a dummies.

# In[19]:


#Para saber qué columnas tienen dos posibles valores, ya que si son categóricas, debemos transformarlas.
for i in range(len(data.columns)):
    etiqueta = data.columns
    if(len(data[etiqueta[i]].unique())== 2):
        print(etiqueta[i])


# El método utilizado para tratar las variables categóricas es One Hot Encoding. Esto detecta cada categoría de cada variable y luego las pasa a diferentes columnas, una para cada una de ellas. Luego se conservan solo una de ellas y el resto son eliminadas, para reducir columnas y que no quede la información duplicada. 

# In[20]:


#transformación de variables categóricas, pasándolas a dummies, mediante el método One Hot Encoding
data = pd.get_dummies(data, columns = ["term"])
data = pd.get_dummies(data, columns = ["pymnt_plan"])
data = pd.get_dummies(data, columns = ["initial_list_status"])
data = pd.get_dummies(data, columns = ["application_type"])
data = pd.get_dummies(data, columns = ["hardship_flag"])
data = pd.get_dummies(data, columns = ["disbursement_method"])
data = pd.get_dummies(data, columns = ["debt_settlement_flag"])
#eliminar las variables duplicadas con las que no nos vamos a quedar
data = data.drop(['term_ 60','pymnt_plan_y','initial_list_status_f', 'application_type_Joint App','hardship_flag_Y','disbursement_method_DirectPay','debt_settlement_flag_N'],axis =1)


# La variable dependiente, "loan_status" tiene más de dos valores, no obstante, hemos considerado que, teniendo en cuenta que lo que nos interesa es predecir si paga o no, los valores que debería devolver tendrían que ser esos mismo. Esto quiere decir que eliminamos el valor "Current" y que los restantes son "Paga" en caso de que pague y "No paga" en caso de que no lo haga.

# In[21]:


# creacion del filtro:
no_current = (data.loan_status !='Current')

# Aplicacion del filtro:
data=data[no_current]

# Resultado Paga/ no paga:
data.loan_status = data.loan_status.map(lambda x:'Paga' if x == 'Fully Paid' else 'No Paga')


# Una vez definido lo anterior, se pasa a variable dummy esta misma, siendo 1 paga y 0 no paga, para poder trabajar con ella. Nuestra variable dependiente pasa a llamarse "loan_status_Paga".

# In[22]:


#Transformamos nuestra nueva variable dependiente, pasándola a dummy
data = pd.get_dummies(data, columns = ["loan_status"])
data = data.drop(['loan_status_No Paga'],axis=1)


# In[23]:


#Reordenar los índices tras eliminar la variable 'Current', ya que han quedado desordenados y no se puede trabajar con ellos.
longitud = len(data)
tic = []
for i in range(longitud):  
    tic.append(i)
data.index = tic


# Además, la variable "home_ownership" tiene 4 categorías, por lo que es necesario tratarla de diferente modo. A cada una de sus cuatro categorías "own", "rent", "morgate" y "any", se les asigna un número, 1, 2, 3 y 4. De esta forma queda codificada la variable.

# In[24]:


#Filtramos cada categoría y le asignamos su nuevo valor.

data['home_ownership'].where(~(data.home_ownership == 'OWN'), other=1, inplace=True) #own es = 1
data['home_ownership'].where(~(data.home_ownership == 'RENT'), other=2, inplace=True) #rent = 2
data['home_ownership'].where(~(data.home_ownership == 'MORTGAGE'), other=3, inplace=True) #mortage = 3
data['home_ownership'].where(~(data.home_ownership == 'ANY'), other=4, inplace=True) #any = 4
    


# ### Matriz de Correlaciones

# Tras haber limpiado el dataset, y no antes, procedemos a realizar una matriz de correlación con las variables restantes, ya limpias. El objetivo es de ello es apoyarnos en ella en la última decisión acerca de las variables con las que se va a trabajar. Por lo tanto, el objetivo principal es el de buscar fuertes correlaciones, con la variable dependiente o entre las propias variables, de tal forma que pudiesen ocasionar algún problema.

# In[25]:


# Correlacion 
corr = numericas.corr()


# In[26]:


#representación de la matriz de correlación
mask = np.triu(np.ones_like(corr, dtype = bool))
cmap = sns.diverging_palette(222, 222, as_cmap = True)
f, ax = plt.subplots(figsize = (11, 9))
sns.heatmap(corr, mask = mask, cmap=cmap, center = 0, square = True, linewidths = .5)


# De matriz de correlación podemos sacar algunas conclusiones. En primer lugar, se aprecian varios grupos de variables que tienen una fuerte correlación entre sí. Estas son "num_actv_bc_tl", "num_bc_sats", "num_il_tl", "num_rev_accts". "Loan_amnt" también parece tener una fuerte correlación con muchas otras variables, al igual que "open_acc", "revol_bal" y "out_prncp". Otro grupo más azul también está compuesto por "total_rec_prncp", "total_pyment" y "total_rec_late_fee". Por último, también podemos prestar atención a otras variables como "chargeoff_within_12_mths".
# 
# De todas estas variables se prescindirá ya que acumulan una correlación exacta. No obstante, comprendemos que no es criterio suficiente, que tengan una alta correlación con otras variables para ser descartadas, pero hemos comprobado además, que muchas de estas variables son post-crédito y que por tanto no nos aportaría informacion buena para nuestro análisis y no son relevantes. Por ello, se podría decir que la matriz nos ha servido para comprobar que algunas de las variables que ya habíamos estudiado con un previo conocimiento del negocio, además tienen una alta correlación con otras.

# ## Eliminación de variables

# Tras haber limpiado la base de datos, teniendo en cuenta valores nulos y duplicados y habiendo pasado todas nuestras funciones, se eliminan variables que tienen poco significado para nuestro análisis.Estas variables han sido cuidadosamente estudiadas, mediante distintas pruebas. Muchas de estas, como se ha comentado anteriormente en la matriz de correlación, hacen alusión al momento en el que ya se ha concedido el crédito, por lo que de ser usadas para nuestros modelos nos llevarían a resultados erróneos, pudiendo ser estos una falsa concesión de crédito. 
# 
# Es necesario tener en cuenta que solo nos interesan las variables en el momento presente en el que se va a valorar el crédito, por lo que todas aquellas que sean futuras o pasadas no nos interesan y no estarían aportando información relevante. Siguiendo este mismo criterio, todas las variables que muestran información del cliente desde hace 24 meses han sido eliminadas y nos hemos quedado con las más reciente, que han sido 2 meses, 6 meses o 12 meses, según procediera.
# 
# 
# 
# 

# In[27]:


#Estas son las columnas finalmente eliminadas, basándonos en las causas expuestas anteriormente

data = data.drop(["hardship_flag_N","debt_settlement_flag_Y", "hardship_flag_N",  'chargeoff_within_12_mths', 
                  "disbursement_method_Cash",  "loan_amnt",  "total_pymnt", 'out_prncp', 'pymnt_plan_n', 
                  'num_sats',  'last_pymnt_amnt', "initial_list_status_w", "tot_hi_cred_lim" , "fico_range_high",
                  "num_bc_tl" , "last_fico_range_high", "open_acc", "total_acc", "total_rec_prncp", "total_rec_int",
                  "installment", "tax_liens", 'num_bc_sats', 'tot_cur_bal', 'num_rev_accts', 'num_bc_sats', 'total_rec_prncp', 
                  'total_rec_late_fee', 'total_rec_int', 'num_tl_90g_dpd_24m', 'open_il_24m', 'open_rv_24m',
                  'acc_open_past_24mths','total_il_high_credit_limit', 'last_fico_range_low', "int_rate" ], axis=1) 


# Por otro lado, las columnas con las que finalmente hemos decidido trabajar son las siguientes, y consideramos que son las que más información aportan a nuestro análisis. Se le ha prestado especial importancia a las variables que pudiesen contener algún tipo de trayectoria de impago del cliente. Además, las columnas "fico_range_high" y "last_fico_range_high" han sido eliminadas y solo se han conservado su versión de low, ya que se considera que suponen un riesgo de impago y un factor importante a tener en cuenta, mientras que la puntuación favorable no tanto.

# In[29]:


data.shape


# ## Guardar el csv limpio
# 

# In[30]:


etiqueta1 = data.columns
clean_csv= data[etiqueta1].copy()
clean_csv.to_csv('../clean.csv', sep = ';')


# In[ ]:




