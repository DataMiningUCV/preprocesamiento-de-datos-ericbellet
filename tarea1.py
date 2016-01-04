# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:37:56 2015
@author: eric
"""
import string
import pylab as pl
import pandas as pd
import csv as csv 
import numpy as np
import os
import matplotlib.pyplot as plt
import xray
import re
from pandas import Series
from pandas import DataFrame
from sklearn import preprocessing

os.getcwd() # obtener dir
os.chdir('/home/eric/Escritorio') # cambiar la direccion

#Esta funcion me dice si un valor es entero o no
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
#------------------------------------------------------------------------------
#REALIZO UN CAMBIO EN EL NOMBRE DE LAS COLUMNAS
data_file = open('data.csv', 'rb')
data_file_object = csv.reader(data_file)
header = data_file_object.next()

#Creo otro .csv identico al original pero con los nombre modificados
data2_file = open("data2.csv", "wb")
data2_file_object = csv.writer(data2_file)
data2_file_object.writerow(["ID", "PeriodoAcademicoarenovar", "CedulaDeIdentidad", "FechadeNacimiento", "Edad", "EstadoCivil", "Sexo", "Escuela", "AnodeIngresoalaUCV", "ModalidaddeIngresoalaUCV", "Semestrequecursa", "Hacambiadousteddedireccion", "Deserafirmativoindiqueelmotivo", "Numerodemateriasinscritasenelsemestreoanoanterior", "Numerodemateriaaprobadasenelsemestreoanoanterior", "Numerodemateriasretiradasenelsemestreoanoanterior", "Numerodemateriasreprobadasenelsemestreoanoanterior", "Promedioponderadoaprobado", "Eficiencia", "Sireprobounaomasmateriasindiqueelmotivo", "Numerodemateriasinscritasenelsemestreencurso", "Seencuentrarealizandotesisopasantiasdegrado", "Cantidaddevecesqueharealizadotesisopasantiasdegrado", "Procedencia", "LugardonderesidemientrasestudiaenlaUniversidad", "Personasconlascualesustedvivemientrasestudiaenlauniversidad", "Tipodeviviendadonderesidemientrasestudiaenlauniversidad", "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual", "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada", "Contrajomatrimonio", "HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion.", "Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo", "Seencuentraustedrealizandoalgunaactividadquelegenereingresos", "Encasodeserafirmativoindiquetipodeactividadysufrecuencia", "Montomensualdelabeca",  "Aportemensualquelebrindasuresponsableeconomico", "Aportemensualquerecibedefamiliaresoamigos", "Ingresomensualquerecibeporactividadesadestajooporhoras", "Ingresomensualtotal", "Gastosenalimentacionpersonal", "Gastosentransportepersonal", "Gastosmedicospersonal", "Gastosodontologicospersonal", "Gastospersonales", "Gastosenresidenciaohabitacionalquiladapersonal", "GastosenMaterialesdeestudiopersonal", "Gastosenrecreacionpersonal", "Otrosgastospersonal", "Totalegresospersonal", "Indiquequienessuresponsableeconomico", "Cargafamiliar", "Ingresomensualdesuresponsableeconomico", "Otrosingresos", "Totaldeingresos", "Gastosenviviendadesusresponsableseconomicos ", "Gastosenalimentaciondesusresponsableseconomicos",  "Gastosentransportedesusresponsableseconomicos", "Gastosmedicosdesusresponsableseconomicos", "Gastosodontologicosdesusresponsableseconomicos", "Gastoseducativosdesusresponsableseconomicos", "Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos", "Condominiodesusresponsableseconomicos", "Otrosgastosdesusresponsableseconomicos", "Totaldeegresosdesusresponsableseconomicos", "DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE", "Sugerenciasyrecomendacionesparamejorarnuestraatencion"])
for row in data_file_object:       # For each row in test.csv
    data2_file_object.writerow(row)    # predict 0
data_file.close()
data2_file.close()
#------------------------------------------------------------------------------
#Defino el DataFrame con todos los datos
df = pd.read_csv("data2.csv",header=0)
"""
# All missing Embarked -> just make them embark from most common place
if len(df.PeriodoAcademicoarenovar[df.PeriodoAcademicoarenovar.isnull() ]) > 0:
    df.PeriodoAcademicoarenovar[ df.PeriodoAcademicoarenovar.isnull() ] = df.PeriodoAcademicoarenovar.dropna().mode().values

Ports = list(enumerate(np.unique(df['PeriodoAcademicoarenovar'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
"""
#df.PeriodoAcademicoarenovar = df.PeriodoAcademicoarenovar.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
#------------------------------------------------------------------------------CEDULA
#CONVIERTO TODA LA COLUMNA CEDULA EN FLOAT
df.CedulaDeIdentidad = df.CedulaDeIdentidad.astype(float)
#------------------------------------------------------------------------------FechadeNacimiento
#------------------------------------------------------------------------------EDAD
#CONVIERTO TODA LA COLUMNA EDAD EN ENTERO
#Busco las edades que no se pueden transformar en Entero y las acomodo
for x in df['Edad']:
   if RepresentsInt(x) == False:
      edades = re.split('[a-z]+', x, flags=re.IGNORECASE)
      df.loc[df.Edad == x, 'Edad'] = edades[0]

 
#Realizo la transformacion en toda la columna.
df.Edad = df.Edad.astype(float)

#Hago una copia del dataframe, para hayar los OUTLIERS
newdf = df.copy()
newdf['x-Mean'] = abs(newdf['Edad'] - newdf['Edad'].mean())
newdf['1.96*std'] = 1.96*newdf['Edad'].std()  
newdf['Outlier'] = abs(newdf['Edad'] - newdf['Edad'].mean()) > 1.96*newdf['Edad'].std()
"""
#Grafico los outliers
idout = newdf['ID'][(newdf['Outlier'] == True)]
edadout = newdf['Edad'][(newdf['Outlier'] == True)]
idnormal = newdf['ID'][(newdf['Outlier'] == False)]
edadnormal = newdf['Edad'][(newdf['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, edadout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, edadnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""
#------------------------------------------------------------------------------EstadoCivil
df['EstadoCivil'] = df['EstadoCivil'].map( {'Soltero (a)': 0, 'Casado (a)': 1, 'Viudo (a)': 2, 'Unido (a)':3} ).astype(int)

#------------------------------------------------------------------------------SEXO
df['Sexo'] = df['Sexo'].map( {'Femenino': 0, 'Masculino': 1} ).astype(int)
#------------------------------------------------------------------------------ESCUELA
df['Escuela'] = df['Escuela'].map( {'Bioanálisis': 0, 'Enfermería': 1} ).astype(int)
#------------------------------------------------------------------------------AnodeIngresoalaUCV
df.AnodeIngresoalaUCV = df.AnodeIngresoalaUCV.astype(float)
#Hago una copia del dataframe, para hayar los OUTLIERS
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['AnodeIngresoalaUCV'] - ndf['AnodeIngresoalaUCV'].mean())
ndf['1.96*std'] = 1.96*ndf['AnodeIngresoalaUCV'].std()  
ndf['Outlier'] = abs(ndf['AnodeIngresoalaUCV'] - ndf['AnodeIngresoalaUCV'].mean()) > 1.96*ndf['AnodeIngresoalaUCV'].std()


"""
#Grafico los outliers
idout = ndf['ID'][(ndf['Outlier'] == True)]
AnodeIngresoalaUCVout = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == True)]
idnormal = ndf['ID'][(ndf['Outlier'] == False)]
AnodeIngresoalaUCVnormal = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, AnodeIngresoalaUCVout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, AnodeIngresoalaUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""

#------------------------------------------------------------------------------ModalidaddeIngresoalaUCV
df['ModalidaddeIngresoalaUCV'] = df['ModalidaddeIngresoalaUCV'].map( {'Convenios Interinstitucionales (nacionales e internacionales)': 0, 'Prueba Interna y/o propedéutico': 1, 'Convenios Internos (Deportistas, artistas, hijos empleados docente y obreros, Samuel Robinson)': 2, 'Asignado OPSU': 3} ).astype(int)
#------------------------------------------------------------------------------Semestrequecursa
for x in df['Semestrequecursa']:
   if RepresentsInt(x) == False:
      semestre = re.split('[a-z]+', x, flags=re.IGNORECASE)
      df.loc[df.Semestrequecursa == x, 'Semestrequecursa'] = semestre[0]
df.Semestrequecursa = df.Semestrequecursa.astype(float)


ndf1 = df.copy()
ndf1['x-Mean'] = abs(ndf1['Semestrequecursa'] - ndf1['Semestrequecursa'].mean())
ndf1['1.96*std'] = 1.96*ndf1['Semestrequecursa'].std()  
ndf1['Outlier'] = abs(ndf1['Semestrequecursa'] - ndf1['Semestrequecursa'].mean()) > 1.96*ndf1['Semestrequecursa'].std()

"""
#Grafico los outliers
idout = ndf1['ID'][(ndf1['Outlier'] == True)]
Semestrequecursaout = ndf1['Semestrequecursa'][(ndf1['Outlier'] == True)]
idnormal = ndf1['ID'][(ndf1['Outlier'] == False)]
Semestrequecursanormal = ndf1['Semestrequecursa'][(ndf1['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, Semestrequecursaout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, Semestrequecursanormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""
#------------------------------------------------------------------------------Hacambiadousteddedireccion
df['Hacambiadousteddedireccion'] = df['Hacambiadousteddedireccion'].map( {'No': 0, 'Si': 1} ).astype(int)
#------------------------------------------------------------------------------Deserafirmativoindiqueelmotivo
df['Deserafirmativoindiqueelmotivo'].fillna('nan', inplace=True)

df['Deserafirmativoindiqueelmotivo'] = df['Deserafirmativoindiqueelmotivo'].map( {'nan': 0,'situacion de salud de un familiar': 1,'Unión a pareja': 2, 'porque mis padres se mudaron': 3,'No me llegan correos de ustedes': 4, 'mudanza': 5,'vivia con mi tia, ahora vivo con mi madre en los valles del tuy': 6, 'olvido de clave': 7,'Por venta de la casa.': 8, 'Enfermedad de mi mamá': 9} ).astype(float)

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreoañoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriasinscritasenelsemestreoanoanterior.astype(float)

ndf2 = df.copy()
ndf2['x-Mean'] = abs(ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].mean())
ndf2['1.96*std'] = 1.96*ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].std()  
ndf2['Outlier'] = abs(ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].mean()) > 1.96*ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].std()

"""

#Grafico los outliers
idout = ndf2['ID'][(ndf2['Outlier'] == True)]
Numerodemateriasinscritasenelsemestreoanoanteriorout = ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'][(ndf2['Outlier'] == True)]
idnormal = ndf2['ID'][(ndf2['Outlier'] == False)]
Numerodemateriasinscritasenelsemestreoanoanteriornormal = ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'][(ndf2['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, Numerodemateriasinscritasenelsemestreoanoanteriorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, Numerodemateriasinscritasenelsemestreoanoanteriornormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""
#------------------------------------------------------------------------------Numerodemateriaaprobadasenelsemestreoanoanterior
for x in df['Numerodemateriaaprobadasenelsemestreoanoanterior']:
   if RepresentsInt(x) == False:
       all=string.maketrans('','')
       nodigs=all.translate(all, string.digits)
       semestre = x.translate(all, nodigs)
       semestree =str(semestre)
       df.loc[df.Numerodemateriaaprobadasenelsemestreoanoanterior == x, 'Numerodemateriaaprobadasenelsemestreoanoanterior'] = semestree

df.Numerodemateriaaprobadasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior.astype(float)

#------------------------------------------------------------------------------Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasretiradasenelsemestreoanoanterior = df.Numerodemateriasretiradasenelsemestreoanoanterior.astype(float)
ndf4 = df.copy()
ndf4['x-Mean'] = abs(ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].mean())
ndf4['1.96*std'] = 1.96*ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].std()  
ndf4['Outlier'] = abs(ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].mean()) > 1.96*ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].std()


"""
#Grafico los outliers
idout = ndf4['ID'][(ndf4['Outlier'] == True)]
Numerodemateriasretiradasenelsemestreoanoanteriorout = ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'][(ndf4['Outlier'] == True)]
idnormal = ndf4['ID'][(ndf4['Outlier'] == False)]
Numerodemateriasretiradasenelsemestreoanoanteriornormal = ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'][(ndf4['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, Numerodemateriasretiradasenelsemestreoanoanteriorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, Numerodemateriasretiradasenelsemestreoanoanteriornormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""
#------------------------------------------------------------------------------Numerodemateriasreprobadasenelsemestreoanoanterior
df.Numerodemateriasreprobadasenelsemestreoanoanterior = df.Numerodemateriasreprobadasenelsemestreoanoanterior.astype(float)

ndf5 = df.copy()
ndf5['x-Mean'] = abs(ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean())
ndf5['1.96*std'] = 1.96*ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()  
ndf5['Outlier'] = abs(ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean()) > 1.96*ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()


"""
#Grafico los outliers
idout = ndf5['ID'][(ndf5['Outlier'] == True)]
Numerodemateriasreprobadasenelsemestreoanoanteriorout = ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'][(ndf5['Outlier'] == True)]
idnormal = ndf5['ID'][(ndf5['Outlier'] == False)]
Numerodemateriasreprobadasenelsemestreoanoanteriornormal = ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'][(ndf5['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, Numerodemateriasreprobadasenelsemestreoanoanteriorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, Numerodemateriasreprobadasenelsemestreoanoanteriornormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""

#------------------------------------------------------------------------------Promedioponderadoaprobado
df.Promedioponderadoaprobado = df.Promedioponderadoaprobado.astype(float)
for x in df['Promedioponderadoaprobado']:
    if x > 100000:
      valor = x/10000
      df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor

    if x >= 10000 and x < 100000:
        valor = x/1000
        
        df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor
    
 
   

#------------------------------------------------------------------------------Eficiencia
for x in df['Eficiencia']:
    if int(x) >1:
        
        valor = "0." + str(int(x))
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor


df.Eficiencia = df.Eficiencia.astype(float)       
    
 
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Eficiencia'] - ndf6['Eficiencia'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Eficiencia'].std()  
ndf6['Outlier'] = abs(ndf6['Eficiencia'] - ndf6['Eficiencia'].mean()) > 1.96*ndf6['Eficiencia'].std()
"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
Eficienciaout = ndf6['Eficiencia'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
Eficiencianormal = ndf6['Eficiencia'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, Eficienciaout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, Eficiencianormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""

#------------------------------------------------------------------------------Sireprobounaomasmateriasindiqueelmotivo
print "ANALIZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR"

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreencurso
df.Numerodemateriasinscritasenelsemestreencurso = df.Numerodemateriasinscritasenelsemestreencurso.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Numerodemateriasinscritasenelsemestreencurso'] - ndf6['Numerodemateriasinscritasenelsemestreencurso'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Numerodemateriasinscritasenelsemestreencurso'].std()  
ndf6['Outlier'] = abs(ndf6['Numerodemateriasinscritasenelsemestreencurso'] - ndf6['Numerodemateriasinscritasenelsemestreencurso'].mean()) > 1.96*ndf6['Numerodemateriasinscritasenelsemestreencurso'].std()
"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Numerodemateriasinscritasenelsemestreencurso'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Numerodemateriasinscritasenelsemestreencurso'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()"""
#------------------------------------------------------------------------------Seencuentrarealizandotesisopasantiasdegrado
df['Seencuentrarealizandotesisopasantiasdegrado'] = df['Seencuentrarealizandotesisopasantiasdegrado'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------Cantidaddevecesqueharealizadotesisopasantiasdegrado

df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].fillna(0, inplace=True)
df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'] = df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].map( {0:0,'Primera vez': 1, 'Segunda vez': 2, 'Más de dos': 3} ).astype(float)

#------------------------------------------------------------------------------Procedencia
df['Procedencia'] = df['Procedencia'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25} ).astype(float)

#------------------------------------------------------------------------------"LugardonderesidemientrasestudiaenlaUniversidad"
#SUstituyo los valores nulos por su procedencia
df['LugardonderesidemientrasestudiaenlaUniversidad'].fillna(0, inplace=True)
df['LugardonderesidemientrasestudiaenlaUniversidad'] = df['LugardonderesidemientrasestudiaenlaUniversidad'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25,0:26} ).astype(float)
contador=0

for x in df['LugardonderesidemientrasestudiaenlaUniversidad']:
    if x == 26:
        valor = df['Procedencia'][contador]
        df.loc[df.LugardonderesidemientrasestudiaenlaUniversidad == x, 'LugardonderesidemientrasestudiaenlaUniversidad'] = valor
        
    contador=contador+1
    
   
#------------------------------------------------------------------------------"Personasconlascualesustedvivemientrasestudiaenlauniversidad"
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] = df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].map( {'residencia estudiantil':0, 'recidencia':0, 'residencia':0,'Residencia':0,'Esposo (a) Hijos (as) ': 1, 'Familiares paternos': 2, 'Madre': 3,'Familiares maternos': 4, 'Ambos padres': 5, 'Mi Mamá y mi hijo ': 6,'Padre': 7, 'Amigos': 8, 'madre y su esposo,abuela,y mi esposo': 9,'hermana': 10, 'Hermana': 10, 'dos hermanos':11, 'OTROS INQUILINOS': 12, 'hermanas':13, 'madre y hermana': 14, 'madre y hermanos':15, 'Madre y Hermanos':15, 'prima': 16, 'madrina':17, 'sola': 18, 'Mamá y Abuela': 19, 'madre,hermano e hijo':20, 'Madre, Hermano y Sobrina': 21, 'hermano, hermana y mi hijo': 22, 'compañeros de habitacion alquilada':23, 'Padres, hermana y abuelos maternos': 24, 'Madre, Hermana, Abuela': 25, 'abuela': 26, 'Dueños del apartamento donde alquilo la habitacion': 27, 'ambos padres y dos hermanis':28, 'Solo': 29, 'hermano':30, 'dueña del apartamento': 31, 'Madre y hermano': 32} ).astype(float)
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad']
#------------------------------------------------------------------------------"Tipodeviviendadonderesidemientrasestudiaenlauniversidad"
#------------------------------------------------------------------------------ "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual"
#------------------------------------------------------------------------------ "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada"
#------------------------------------------------------------------------------"Contrajomatrimonio"
#------------------------------------------------------------------------------"HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion."
#------------------------------------------------------------------------------"Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo"
#------------------------------------------------------------------------------ "Seencuentraustedrealizandoalgunaactividadquelegenereingresos"
 #------------------------------------------------------------------------------"Encasodeserafirmativoindiquetipodeactividadysufrecuencia"
 #------------------------------------------------------------------------------"Montomensualdelabeca"
 #------------------------------------------------------------------------------ "Aportemensualquelebrindasuresponsableeconomico"
 #------------------------------------------------------------------------------"Aportemensualquerecibedefamiliaresoamigos"
 #------------------------------------------------------------------------------"Ingresomensualquerecibeporactividadesadestajooporhoras"
 #------------------------------------------------------------------------------"Ingresomensualtotal"
 #------------------------------------------------------------------------------"Gastosenalimentacionpersonal"
 #------------------------------------------------------------------------------"Gastosentransportepersonal"
 #------------------------------------------------------------------------------"Gastosmedicospersonal"
 #------------------------------------------------------------------------------"Gastosodontologicospersonal"
 #------------------------------------------------------------------------------"Gastospersonales"
 #------------------------------------------------------------------------------"Gastosenresidenciaohabitacionalquiladapersonal"
 #------------------------------------------------------------------------------"GastosenMaterialesdeestudiopersonal"
 #------------------------------------------------------------------------------"Gastosenrecreacionpersonal"
 #------------------------------------------------------------------------------"Otrosgastospersonal"
 #------------------------------------------------------------------------------"Totalegresospersonal"
 #------------------------------------------------------------------------------"Indiquequienessuresponsableeconomico"
 #------------------------------------------------------------------------------"Cargafamiliar"
 #------------------------------------------------------------------------------"Ingresomensualdesuresponsableeconomico"
 #------------------------------------------------------------------------------"Otrosingresos"
 #------------------------------------------------------------------------------"Totaldeingresos"
 #------------------------------------------------------------------------------"Gastosenviviendadesusresponsableseconomicos"
 #------------------------------------------------------------------------------"Gastosenalimentaciondesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Gastosentransportedesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Gastosmedicosdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Gastosodontologicosdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Gastoseducativosdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Condominiodesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Otrosgastosdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"Totaldeegresosdesusresponsableseconomicos"
  #------------------------------------------------------------------------------"DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE"
 #------------------------------------------------------------------------------"Sugerenciasyrecomendacionesparamejorarnuestraatencion"


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------


#Genero el .cvs a partir del dataframe
#df.to_csv("minable.csv", sep='\t')
