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
from sklearn.preprocessing import Imputer
from scipy.stats import mode
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
        
def RepresentsFloat(s):
    try: 
        float(s)
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
data2_file_object.writerow(["ID", "PeriodoAcademicoarenovar", "CedulaDeIdentidad", "FechadeNacimiento", "Edad", "EstadoCivil", "Sexo", "Escuela", "AnodeIngresoalaUCV", "ModalidaddeIngresoalaUCV", "Semestrequecursa", "Hacambiadousteddedireccion", "Deserafirmativoindiqueelmotivo", "Numerodemateriasinscritasenelsemestreoanoanterior", "Numerodemateriaaprobadasenelsemestreoanoanterior", "Numerodemateriasretiradasenelsemestreoanoanterior", "Numerodemateriasreprobadasenelsemestreoanoanterior", "Promedioponderadoaprobado", "Eficiencia", "Sireprobounaomasmateriasindiqueelmotivo", "Numerodemateriasinscritasenelsemestreencurso", "Seencuentrarealizandotesisopasantiasdegrado", "Cantidaddevecesqueharealizadotesisopasantiasdegrado", "Procedencia", "LugardonderesidemientrasestudiaenlaUniversidad", "Personasconlascualesustedvivemientrasestudiaenlauniversidad", "Tipodeviviendadonderesidemientrasestudiaenlauniversidad", "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual", "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada", "Contrajomatrimonio", "HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion", "Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo", "Seencuentraustedrealizandoalgunaactividadquelegenereingresos", "Encasodeserafirmativoindiquetipodeactividadysufrecuencia", "Montomensualdelabeca",  "Aportemensualquelebrindasuresponsableeconomico", "Aportemensualquerecibedefamiliaresoamigos", "Ingresomensualquerecibeporactividadesadestajooporhoras", "Ingresomensualtotal", "Gastosenalimentacionpersonal", "Gastosentransportepersonal", "Gastosmedicospersonal", "Gastosodontologicospersonal", "Gastospersonales", "Gastosenresidenciaohabitacionalquiladapersonal", "GastosenMaterialesdeestudiopersonal", "Gastosenrecreacionpersonal", "Otrosgastospersonal", "Totalegresospersonal", "Indiquequienessuresponsableeconomico", "Cargafamiliar", "Ingresomensualdesuresponsableeconomico", "Otrosingresos", "Totaldeingresos", "Gastosenviviendadesusresponsableseconomicos", "Gastosenalimentaciondesusresponsableseconomicos",  "Gastosentransportedesusresponsableseconomicos", "Gastosmedicosdesusresponsableseconomicos", "Gastosodontologicosdesusresponsableseconomicos", "Gastoseducativosdesusresponsableseconomicos", "Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos", "Condominiodesusresponsableseconomicos", "Otrosgastosdesusresponsableseconomicos", "Totaldeegresosdesusresponsableseconomicos", "DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE", "Sugerenciasyrecomendacionesparamejorarnuestraatencion"])
for row in data_file_object:       # For each row in test.csv
    data2_file_object.writerow(row)    # predict 0
data_file.close()
data2_file.close()
#------------------------------------------------------------------------------

#Defino el DataFrame con todos los datos
df = pd.read_csv("data2.csv",header=0)

#------------------------------------------------------------------------------ID
df.ID = df.ID.astype(int)
#------------------------------------------------------------------------------PeriodoAcademicoarenovar
df['AnoAcademicoarenovar'] = df['PeriodoAcademicoarenovar']
df['SemestreAcademicoarenovar'] =  df['PeriodoAcademicoarenovar']
for x in df['PeriodoAcademicoarenovar']:
    #print periodo
    periodo =  re.findall('[a-zA-Z]+|\\d+', x)
    tam = len(periodo)
    
   
    if tam == 2 or tam == 5:
        if periodo[1] == "01" or periodo[1] == "1" or  periodo[0] == "pri" or periodo[0] == "Pri" or periodo[0] == "PRI" or periodo[0] == "1" or periodo[0] == "prim":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "I"
        if periodo[1] == "02" or periodo[1] == "2" or periodo[1] == "II" or periodo[0] == "seg" or periodo[0] == "Seg" or periodo[0] == "segundo" or periodo[0] == "2" or periodo[0] == "SEG" or periodo[0] == "Segundo" or periodo[0] == "sec":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "II"
        if periodo[1] == "2015" or periodo[1] == "15" or periodo[0] == "2015" or periodo[0] == "15":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2015" 
        if periodo[1] == "2014" or periodo[1] == "14" or periodo[0] == "2014" or periodo[0] == "14":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2014" 
        if periodo[0] == "I" or periodo[0] == "II":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = periodo[0]
            
            
    if tam == 3:
        if periodo[2] == "1" or periodo[0] == "pri" or periodo[0] == "Pri" or periodo[0] == "PRI" or periodo[0] == "1" or periodo[0] == "prim" or periodo[1] == "01" or periodo[1] == "1":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "I"
        if periodo[2] == "2015" or periodo[2] == "15" or periodo[0] == "2015" or periodo[0] == "15":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2015" 
        if periodo[2] == "2014" or periodo[2] == "14" or periodo[0] == "2014" or periodo[0] == "14":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2014"
        if periodo[0] == "I" or periodo[0] == "II":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = periodo[0]
        if periodo[1] == "02" or periodo[1] == "2" or periodo[1] == "II" or periodo[0] == "seg" or periodo[0] == "Seg" or periodo[0] == "segundo" or periodo[0] == "2" or periodo[0] == "SEG" or periodo[0] == "Segundo" or periodo[0] == "sec":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "II"
            
    if tam == 1:
        if periodo[0] == "I" or periodo[0] == "II":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = periodo[0]
            
        if periodo[0] == "pri" or periodo[0] == "Pri" or periodo[0] == "PRI" or periodo[0] == "1" or periodo[0] == "prim":
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "I"
          
        if periodo[0] == "seg" or periodo[0] == "Seg" or periodo[0] == "segundo" or periodo[0] == "2" or periodo[0] == "SEG" or periodo[0] == "Segundo" or periodo[0] == "sec":  
            df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "II"
            
        if periodo[0] == "2015" or periodo[0] == "15":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2015" 
        if periodo[0] == "2014" or periodo[0] == "14":
            df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "2014"


for x in df.AnoAcademicoarenovar:
    if x != "2015" and x != "2014":
         df.loc[df.AnoAcademicoarenovar == x, 'AnoAcademicoarenovar'] = "NaN" 
         
for x in df.SemestreAcademicoarenovar:
    if x != "I" and x != "II":
         df.loc[df.SemestreAcademicoarenovar == x, 'SemestreAcademicoarenovar'] = "NaN" 
         
df['SemestreAcademicoarenovar'] = df['SemestreAcademicoarenovar'].map( {'I': 1, 'II': 2} ) 
df.AnoAcademicoarenovar = df.AnoAcademicoarenovar.astype(float)
x = mode(df['AnoAcademicoarenovar'] )
y = mode(df['SemestreAcademicoarenovar'] )

df['AnoAcademicoarenovar'].fillna(int(x[0]), inplace=True)
df['SemestreAcademicoarenovar'].fillna(int(y[0]), inplace=True)

df.AnoAcademicoarenovar = df.AnoAcademicoarenovar.astype(int)   
df.SemestreAcademicoarenovar = df.SemestreAcademicoarenovar.astype(int)  
#------------------------------------------------------------------------------CEDULA
#CONVIERTO TODA LA COLUMNA CEDULA EN FLOAT
df.CedulaDeIdentidad = df.CedulaDeIdentidad.astype(float)
#------------------------------------------------------------------------------FechadeNacimiento
df['DiadeNacimiento'] = df['FechadeNacimiento']
df['MesdeNacimiento'] = df['FechadeNacimiento']
df['AnodeNacimiento'] = df['FechadeNacimiento']


for x in df['FechadeNacimiento']:
    #print periodo
    fecha =  re.findall('[a-zA-Z]+|\\d+', x)
    #fecha
    tam = len(fecha)
    
    if tam == 1:
        string = fecha[0]
        firstpart, secondpart = string[:len(string)/2], string[len(string)/2:]
        if int(secondpart) > 1900:
            df.loc[df.AnodeNacimiento == x, 'AnodeNacimiento'] = secondpart
            string = firstpart
            firstpart, secondpart = string[:len(string)/2], string[len(string)/2:]
            df.loc[df.DiadeNacimiento == x, 'DiadeNacimiento'] = firstpart
            df.loc[df.MesdeNacimiento == x, 'MesdeNacimiento'] = secondpart
        if string == "19220485":
            df.loc[df.DiadeNacimiento == x, 'DiadeNacimiento'] = "22"
            df.loc[df.MesdeNacimiento == x, 'MesdeNacimiento'] = "04"
            df.loc[df.AnodeNacimiento == x, 'AnodeNacimiento'] = "1985"
        
        
    if tam == 3:
        df.loc[df.DiadeNacimiento == x, 'DiadeNacimiento'] = fecha[0]
        df.loc[df.MesdeNacimiento == x, 'MesdeNacimiento'] = fecha[1]
        
        if len(fecha[2]) == 4:
            if int(fecha[2]) > 2000:
                df.loc[df.AnodeNacimiento == x, 'AnodeNacimiento'] = "1991"
            else:
                df.loc[df.AnodeNacimiento == x, 'AnodeNacimiento'] = fecha[2]
            
                
        
        if len(fecha[2]) == 2:
            df.loc[df.AnodeNacimiento == x, 'AnodeNacimiento'] = "19" +fecha[2]
#Elimino esta fila ya que la fecha esta completamente incorrecta   
#df = df.drop(df.index[19])      
       
          
df.DiadeNacimiento = df.DiadeNacimiento.astype(int)
          
df.MesdeNacimiento = df.MesdeNacimiento.astype(int)

df.AnodeNacimiento = df.AnodeNacimiento.astype(int)
      
   
         


#------------------------------------------------------------------------------EDAD
#CONVIERTO TODA LA COLUMNA EDAD EN ENTERO
#Busco las edades que no se pueden transformar en Entero y las acomodo
for x in df['Edad']:
   if RepresentsInt(x) == False:
      edades = re.split('[a-z]+', x, flags=re.IGNORECASE)
      df.loc[df.Edad == x, 'Edad'] = edades[0]

 
#Realizo la transformacion en toda la columna.
df.Edad = df.Edad.astype(int)

#Hago una copia del dataframe, para hayar los OUTLIERS
newdf = df.copy()
newdf['x-Mean'] = abs(newdf['Edad'] - newdf['Edad'].mean())
newdf['1.96*std'] = 1.96*newdf['Edad'].std()  
newdf['Outlier'] = abs(newdf['Edad'] - newdf['Edad'].mean()) > 1.96*newdf['Edad'].std()

# data 
idout = newdf['ID'][(newdf['Outlier'] == True)]
edadout = newdf['Edad'][(newdf['Outlier'] == True)]
idnormal = newdf['ID'][(newdf['Outlier'] == False)]
edadnormal = newdf['Edad'][(newdf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('ID')
plt.ylabel('Edad')
plt.title('Outliers: Edades de los estudiantes')
out = plt.scatter(idout, edadout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, edadnormal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Edad.png')


#------------------------------------------------------------------------------EstadoCivil

df['EstadoCivil'] = df['EstadoCivil'].map( {'Soltero (a)': 0, 'Casado (a)': 1, 'Viudo (a)': 2, 'Unido (a)':3} ).astype(int)

#------------------------------------------------------------------------------SEXO
df['Sexo'] = df['Sexo'].map( {'Femenino': 0, 'Masculino': 1} ).astype(int)

#------------------------------------------------------------------------------ESCUELA
df['Escuela'] = df['Escuela'].map( {'Bioanálisis': 0, 'Enfermería': 1} ).astype(int)

#------------------------------------------------------------------------------AnodeIngresoalaUCV
df.AnodeIngresoalaUCV = df.AnodeIngresoalaUCV.astype(int)
#Hago una copia del dataframe, para hayar los OUTLIERS
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['AnodeIngresoalaUCV'] - ndf['AnodeIngresoalaUCV'].mean())
ndf['1.96*std'] = 1.96*ndf['AnodeIngresoalaUCV'].std()  
ndf['Outlier'] = abs(ndf['AnodeIngresoalaUCV'] - ndf['AnodeIngresoalaUCV'].mean()) > 1.96*ndf['AnodeIngresoalaUCV'].std()



#Grafico los outliers
idout = ndf['ID'][(ndf['Outlier'] == True)]
AnodeIngresoalaUCVout = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == True)]
idnormal = ndf['ID'][(ndf['Outlier'] == False)]
AnodeIngresoalaUCVnormal = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('ID')
plt.ylabel('Ano de Ingreso a la UCV')
plt.title('Outliers: Ano de Ingreso a la UCV de los estudiantes')
out = plt.scatter(idout, AnodeIngresoalaUCVout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, AnodeIngresoalaUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('AnodeIngresoalaUCV.png')


#------------------------------------------------------------------------------ModalidaddeIngresoalaUCV
df['ModalidaddeIngresoalaUCV'] = df['ModalidaddeIngresoalaUCV'].map( {'Convenios Interinstitucionales (nacionales e internacionales)': 0, 'Prueba Interna y/o propedéutico': 1, 'Convenios Internos (Deportistas, artistas, hijos empleados docente y obreros, Samuel Robinson)': 2, 'Asignado OPSU': 3} ).astype(int)

#------------------------------------------------------------------------------Semestrequecursa
for x in df['Semestrequecursa']:
   if RepresentsInt(x) == False:
      semestre = re.split('[a-z]+', x, flags=re.IGNORECASE)
      df.loc[df.Semestrequecursa == x, 'Semestrequecursa'] = semestre[0]
df.Semestrequecursa = df.Semestrequecursa.astype(int)


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
#mode(df['Hacambiadousteddedireccion'] )
#------------------------------------------------------------------------------Deserafirmativoindiqueelmotivo
df['Deserafirmativoindiqueelmotivo'].fillna('nan', inplace=True)

df['Deserafirmativoindiqueelmotivo'] = df['Deserafirmativoindiqueelmotivo'].map( {'nan': 0,'situacion de salud de un familiar': 1,'Unión a pareja': 2, 'porque mis padres se mudaron': 2,'No me llegan correos de ustedes': 3, 'mudanza': 2, 'mudanza ': 2,'vivia con mi tia, ahora vivo con mi madre en los valles del tuy': 2, 'olvido de clave': 7,'Por venta de la casa.': 2, 'Enfermedad de mi mamá': 1} ).astype(float)
df['Deserafirmativoindiqueelmotivo'] = df['Deserafirmativoindiqueelmotivo'].astype(int)

#respondieron = df.loc[df['Deserafirmativoindiqueelmotivo'] != 0]
#mode(respondieron['Deserafirmativoindiqueelmotivo'])
#------------------------------------------------------------------------------Numerodemateriaaprobadasenelsemestreoanoanterior
for x in df['Numerodemateriaaprobadasenelsemestreoanoanterior']:
   if RepresentsInt(x) == False:
      num = re.split('[a-z]+', x, flags=re.IGNORECASE)
      num
      df.loc[df.Numerodemateriaaprobadasenelsemestreoanoanterior == x, 'Numerodemateriaaprobadasenelsemestreoanoanterior'] = num[3]

df.Numerodemateriaaprobadasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior.astype(int)

#------------------------------------------------------------------------------Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasretiradasenelsemestreoanoanterior = df.Numerodemateriasretiradasenelsemestreoanoanterior.astype(int)
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
df.Numerodemateriasreprobadasenelsemestreoanoanterior = df.Numerodemateriasreprobadasenelsemestreoanoanterior.astype(int)

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

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreoañoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior + df.Numerodemateriasreprobadasenelsemestreoanoanterior + df.Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriasinscritasenelsemestreoanoanterior.astype(int)

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
#------------------------------------------------------------------------------Promedioponderadoaprobado
df.Promedioponderadoaprobado = df.Promedioponderadoaprobado.astype(float)
for x in df['Promedioponderadoaprobado']:
    if x > 100000:
      valor = x/10000
      df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor

    if x >= 10000 and x < 100000:
        valor = x/1000
        
        df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor
    
df['Promedioponderadoaprobado'] = df['Promedioponderadoaprobado'].round(2) 
   

#------------------------------------------------------------------------------Eficiencia
df['Eficiencia'].round(2)
for x in df['Eficiencia']:
    
    if int(x) >1:
        
        valor = "0." + str(int(x))
        
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor

df.Eficiencia = df.Eficiencia.astype(float)
df['Eficiencia'] = df['Eficiencia'].round(2) 
    
 
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

Ports = list(enumerate(np.unique(df['Sireprobounaomasmateriasindiqueelmotivo'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Sireprobounaomasmateriasindiqueelmotivo = df.Sireprobounaomasmateriasindiqueelmotivo.map( lambda x: Ports_dict[x]).astype(int)    

    

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreencurso
df.Numerodemateriasinscritasenelsemestreencurso = df.Numerodemateriasinscritasenelsemestreencurso.astype(int)

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
df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'] = df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].astype(int)
#------------------------------------------------------------------------------Procedencia
df['Procedencia'] = df['Procedencia'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25} ).astype(float)
df['Procedencia'] = df['Procedencia'].astype(int)
#------------------------------------------------------------------------------"LugardonderesidemientrasestudiaenlaUniversidad"
#SUstituyo los valores nulos por su procedencia
df['LugardonderesidemientrasestudiaenlaUniversidad'].fillna(0, inplace=True)
df['LugardonderesidemientrasestudiaenlaUniversidad'] = df['LugardonderesidemientrasestudiaenlaUniversidad'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25,0:26} ).astype(float)
contador=0

for x in df['LugardonderesidemientrasestudiaenlaUniversidad']:
    if x == 26:
        valor = df['Procedencia'][contador]    
        df['LugardonderesidemientrasestudiaenlaUniversidad'][contador] = valor
        
    contador=contador+1
    
df['LugardonderesidemientrasestudiaenlaUniversidad'] = df['LugardonderesidemientrasestudiaenlaUniversidad'].astype(int)   
#------------------------------------------------------------------------------"Personasconlascualesustedvivemientrasestudiaenlauniversidad"
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] = df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].map( {'residencia estudiantil':0, 'recidencia':0, 'residencia':0,'Residencia':0,'Esposo (a) Hijos (as) ': 1,'Esposo (a) Hijos (as)\xc2\xa0': 1, 'Familiares paternos': 2, 'Madre': 3,'Familiares maternos': 4, 'Ambos padres': 5, 'Mi Mamá y mi hijo ': 6,'Padre': 7, 'Amigos': 8, 'madre y su esposo,abuela,y mi esposo': 9,'hermana': 10, 'Hermana': 10, 'dos hermanos':11, 'OTROS INQUILINOS': 12, 'hermanas':13, 'madre y hermana': 14, 'madre y hermanos':15, 'Madre y Hermanos':15, 'prima': 16, 'madrina':17, 'sola': 18, 'Mamá y Abuela': 19, 'madre,hermano e hijo':20, 'Madre, Hermano y Sobrina': 21, 'hermano, hermana y mi hijo': 22, 'compañeros de habitacion alquilada':23, 'Padres, hermana y abuelos maternos': 24, 'Madre, Hermana, Abuela': 25, 'abuela': 26, 'Dueños del apartamento donde alquilo la habitacion': 27, 'ambos padres y dos hermanis':28, 'Solo': 29, 'hermano':30, 'dueña del apartamento': 31, 'Madre y hermano': 32} ).astype(float)
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] = df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].astype(int) 
#------------------------------------------------------------------------------"Tipodeviviendadonderesidemientrasestudiaenlauniversidad"
df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] = df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].map( {'Quinta o casa quinta':0, 'Apartamento en edifico': 1, 'Casa en barrio urbano': 2, 'Habitación alquilada': 3,'Casa de vecindad': 4, 'Residencia estudiantil': 5, 'Apartamento en quinta - casa quinta o casa': 6,'conserjería ': 7, 'Casa en barrio rural': 8, 'Casa de vecindad': 9,'casa': 10} ).astype(float)
df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] = df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].astype(int) 
#------------------------------------------------------------------------------ "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual"
#Los que tienen 0 significa que no viven en alquiler o residencia estudiantil o no respondieron
df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].fillna(0, inplace=True)
for x in df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual']:
    if x == "1 UT":
        df.loc[df.Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual == x, 'Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = "125"

    if RepresentsInt(x) == False:
        precio = re.split('[a-z]+', x, flags=re.IGNORECASE)
        df.loc[df.Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual == x, 'Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = precio[0]
df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].astype(float)
#------------------------------------------------------------------------------ "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada"

Ports = list(enumerate(np.unique(df['Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada = df.Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada.map( lambda x: Ports_dict[x]).astype(int)    

#------------------------------------------------------------------------------"Contrajomatrimonio"
df['Contrajomatrimonio'] = df['Contrajomatrimonio'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion."
df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion'] = df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo"
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].fillna(0, inplace=True)
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'] = df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].map( {0:0,'2012, OBE. AYUDA ECONÓMICA. SOLVENTAR GASTOS DE ESTUDIOS Y TRANSPORTE.':13, 'OBE 2011 UCV . Motivo: ayuda económica para mis estudios ': 1, 'AÑO 2011 AYUDA ECONÓMICA POR OBE PARA CUBRIR GASTOS  DE MATERIAL DE ESTUDIO.': 2, 'Beca ayudantía en el año 2011 Universidad Central de Venezuela (OBE) apoyo económico para material de estudio': 3,'solicitud en 2013. \nayuda economica \n': 4, 'Año solicitud 2013\nInstitución: Universidad Central de Venezuela (OBE)\nMotivo: Transporte, Materiales Estudios, Comida': 5, 'fames ayuda para maternidad': 6,'2015, ucv, falta de ingresos, ayuda economica': 7, 'Año: 2013. Institucion: OBE. Motivo: ayuda económica para rehabilitación (fisioterapia), ya que me atropello una moto.': 8, 'Año:2014 \nInstitucion: OBE\nMotivo: Compra de materiales de estudio. ': 9,'fecha:2014\ninstituto:OBE\ncompras de material de estudio': 10,'solicitud 2015 en el instituto OBE , motivo ayuda economica para gastos de la universidad (libro, pasaje, equipo de enfermeria, etc)': 11,'año 2014, OBE, solicitada para cubrir parte de los gastos mensuales estudiantiles': 12} ).astype(float)
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'] = df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].astype(int)
#------------------------------------------------------------------------------ "Seencuentraustedrealizandoalgunaactividadquelegenereingresos"
df['Seencuentraustedrealizandoalgunaactividadquelegenereingresos'] = df['Seencuentraustedrealizandoalgunaactividadquelegenereingresos'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"Encasodeserafirmativoindiquetipodeactividadysufrecuencia"
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].fillna(0, inplace=True)
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'] = df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].map( {0:0,'cuido pacientes parapoder obtener un ingreso economico': 1, 'STAFF DE EMPRESA DEPORTAIVA. UNO O DOS FINES DE SEMANA AL MES APROXIMADAMENTE.': 2, 'Recreación, algunos fines de semana (poco frecuente por falta de tiempo).': 3,'Secretaria, sólo los sábados medio turno.': 4} ).astype(float)
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'] = df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].astype(int)

#------------------------------------------------------------------------------"Montomensualdelabeca"
df.Montomensualdelabeca = df.Montomensualdelabeca.astype(float)
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Montomensualdelabeca'] - ndf6['Montomensualdelabeca'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Montomensualdelabeca'].std()  
ndf6['Outlier'] = abs(ndf6['Montomensualdelabeca'] - ndf6['Montomensualdelabeca'].mean()) > 1.96*ndf6['Montomensualdelabeca'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Montomensualdelabeca'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Montomensualdelabeca'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""
#------------------------------------------------------------------------------ "Aportemensualquelebrindasuresponsableeconomico"
df['Aportemensualquelebrindasuresponsableeconomico'].fillna(0, inplace=True)
df.Aportemensualquelebrindasuresponsableeconomico = df.Aportemensualquelebrindasuresponsableeconomico.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Aportemensualquelebrindasuresponsableeconomico'] - ndf6['Aportemensualquelebrindasuresponsableeconomico'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Aportemensualquelebrindasuresponsableeconomico'].std()  
ndf6['Outlier'] = abs(ndf6['Aportemensualquelebrindasuresponsableeconomico'] - ndf6['Aportemensualquelebrindasuresponsableeconomico'].mean()) > 1.96*ndf6['Aportemensualquelebrindasuresponsableeconomico'].std()
"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Aportemensualquelebrindasuresponsableeconomico'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Aportemensualquelebrindasuresponsableeconomico'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""
#------------------------------------------------------------------------------"Aportemensualquerecibedefamiliaresoamigos"
df['Aportemensualquerecibedefamiliaresoamigos'].fillna(0, inplace=True)
df.Aportemensualquerecibedefamiliaresoamigos = df.Aportemensualquerecibedefamiliaresoamigos.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Aportemensualquerecibedefamiliaresoamigos'] - ndf6['Aportemensualquerecibedefamiliaresoamigos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Aportemensualquerecibedefamiliaresoamigos'].std()  
ndf6['Outlier'] = abs(ndf6['Aportemensualquerecibedefamiliaresoamigos'] - ndf6['Aportemensualquerecibedefamiliaresoamigos'].mean()) > 1.96*ndf6['Aportemensualquerecibedefamiliaresoamigos'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Aportemensualquerecibedefamiliaresoamigos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Aportemensualquerecibedefamiliaresoamigos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""

#------------------------------------------------------------------------------"Ingresomensualquerecibeporactividadesadestajooporhoras"

df['Ingresomensualquerecibeporactividadesadestajooporhoras'].fillna(0, inplace=True)
df.Ingresomensualquerecibeporactividadesadestajooporhoras = df.Ingresomensualquerecibeporactividadesadestajooporhoras.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'] - ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].std()  
ndf6['Outlier'] = abs(ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'] - ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].mean()) > 1.96*ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""
#------------------------------------------------------------------------------"Ingresomensualtotal"
df['Ingresomensualtotal'] = df.Montomensualdelabeca + df.Aportemensualquelebrindasuresponsableeconomico + df.Aportemensualquerecibedefamiliaresoamigos + df.Ingresomensualquerecibeporactividadesadestajooporhoras


#------------------------------------------------------------------------------"Gastosenalimentacionpersonal"
df['Gastosenalimentacionpersonal'].fillna(0, inplace=True)
df.Gastosenalimentacionpersonal = df.Gastosenalimentacionpersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenalimentacionpersonal'] - ndf6['Gastosenalimentacionpersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenalimentacionpersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenalimentacionpersonal'] - ndf6['Gastosenalimentacionpersonal'].mean()) > 1.96*ndf6['Gastosenalimentacionpersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenalimentacionpersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenalimentacionpersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""
#------------------------------------------------------------------------------"Gastosentransportepersonal"
df['Gastosentransportepersonal'].fillna(0, inplace=True)
df.Gastosentransportepersonal = df.Gastosentransportepersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosentransportepersonal'] - ndf6['Gastosentransportepersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosentransportepersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosentransportepersonal'] - ndf6['Gastosentransportepersonal'].mean()) > 1.96*ndf6['Gastosentransportepersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosentransportepersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosentransportepersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Normal student'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.figure()
"""
#------------------------------------------------------------------------------"Gastosmedicospersonal"
df['Gastosmedicospersonal'].fillna(0, inplace=True)
df.Gastosmedicospersonal = df.Gastosmedicospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosmedicospersonal'] - ndf6['Gastosmedicospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosmedicospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosmedicospersonal'] - ndf6['Gastosmedicospersonal'].mean()) > 1.96*ndf6['Gastosmedicospersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosmedicospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosmedicospersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Gastosodontologicospersonal"
df['Gastosodontologicospersonal'].fillna(0, inplace=True)
df.Gastosodontologicospersonal = df.Gastosodontologicospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosodontologicospersonal'] - ndf6['Gastosodontologicospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosodontologicospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosodontologicospersonal'] - ndf6['Gastosodontologicospersonal'].mean()) > 1.96*ndf6['Gastosodontologicospersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosodontologicospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosodontologicospersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Gastospersonales"
df['Gastospersonales'].fillna(0, inplace=True)
df.Gastospersonales = df.Gastospersonales.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastospersonales'] - ndf6['Gastospersonales'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastospersonales'].std()  
ndf6['Outlier'] = abs(ndf6['Gastospersonales'] - ndf6['Gastospersonales'].mean()) > 1.96*ndf6['Gastospersonales'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastospersonales'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastospersonales'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Gastosenresidenciaohabitacionalquiladapersonal"
df['Gastosenresidenciaohabitacionalquiladapersonal'].fillna(0, inplace=True)
df.Gastosenresidenciaohabitacionalquiladapersonal = df.Gastosenresidenciaohabitacionalquiladapersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenresidenciaohabitacionalquiladapersonal'] - ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenresidenciaohabitacionalquiladapersonal'] - ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].mean()) > 1.96*ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenresidenciaohabitacionalquiladapersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenresidenciaohabitacionalquiladapersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"GastosenMaterialesdeestudiopersonal"
df['GastosenMaterialesdeestudiopersonal'].fillna(0, inplace=True)
df.GastosenMaterialesdeestudiopersonal = df.GastosenMaterialesdeestudiopersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['GastosenMaterialesdeestudiopersonal'] - ndf6['GastosenMaterialesdeestudiopersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['GastosenMaterialesdeestudiopersonal'].std()  
ndf6['Outlier'] = abs(ndf6['GastosenMaterialesdeestudiopersonal'] - ndf6['GastosenMaterialesdeestudiopersonal'].mean()) > 1.96*ndf6['GastosenMaterialesdeestudiopersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['GastosenMaterialesdeestudiopersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['GastosenMaterialesdeestudiopersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Gastosenrecreacionpersonal"
df['Gastosenrecreacionpersonal'].fillna(0, inplace=True)
df.Gastosenrecreacionpersonal = df.Gastosenrecreacionpersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenrecreacionpersonal'] - ndf6['Gastosenrecreacionpersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenrecreacionpersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenrecreacionpersonal'] - ndf6['Gastosenrecreacionpersonal'].mean()) > 1.96*ndf6['Gastosenrecreacionpersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenrecreacionpersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenrecreacionpersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Otrosgastospersonal"
df['Otrosgastospersonal'].fillna(0, inplace=True)
df.Otrosgastospersonal = df.Otrosgastospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Otrosgastospersonal'] - ndf6['Otrosgastospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Otrosgastospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Otrosgastospersonal'] - ndf6['Otrosgastospersonal'].mean()) > 1.96*ndf6['Otrosgastospersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Otrosgastospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Otrosgastospersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Totalegresospersonal"
df['Totalegresospersonal'] = df.Gastosenalimentacionpersonal + df.Gastosentransportepersonal + df.Gastosmedicospersonal + df.Gastosodontologicospersonal + df.Gastospersonales + df.Gastosenresidenciaohabitacionalquiladapersonal + df.GastosenMaterialesdeestudiopersonal + df.Gastosenrecreacionpersonal + df.Otrosgastospersonal
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Totalegresospersonal'] - ndf6['Totalegresospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Totalegresospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Totalegresospersonal'] - ndf6['Totalegresospersonal'].mean()) > 1.96*ndf6['Totalegresospersonal'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Totalegresospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Totalegresospersonal'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Indiquequienessuresponsableeconomico"
df['Indiquequienessuresponsableeconomico'] = df['Indiquequienessuresponsableeconomico'].map( {'Usted mismo':0,'esposo': 1, 'ninguno': 2, 'Madre': 3,'Familiares': 4, 'Padre': 5, 'Ambos padres': 6,'Cónyugue': 7, 'tia': 8, 'Hermano': 9,'hermana': 10,'MI HERMANA': 10, 'abuela': 10, 'dos hermanos':11, 'OTROS INQUILINOS': 12, 'hermanas':13, 'madre y hermana': 14, 'madre y hermanos':15, 'Madre y Hermanos':15, 'prima': 16, 'madrina':17, 'sola': 18, 'Mamá y Abuela': 19, 'madre,hermano e hijo':20, 'Madre, Hermano y Sobrina': 21, 'hermano, hermana y mi hijo': 22, 'compañeros de habitacion alquilada':23, 'Padres, hermana y abuelos maternos': 24, 'Madre, Hermana, Abuela': 25, 'abuela': 26, 'Dueños del apartamento donde alquilo la habitacion': 27, 'ambos padres y dos hermanis':28, 'Solo': 29, 'hermano':30, 'dueña del apartamento': 31, 'Madre y hermano': 32} ).astype(float)
df['Indiquequienessuresponsableeconomico'] = df['Indiquequienessuresponsableeconomico'].astype(int)
#------------------------------------------------------------------------------"Cargafamiliar"
df.Cargafamiliar = df.Cargafamiliar.astype(int)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Cargafamiliar'] - ndf6['Cargafamiliar'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Cargafamiliar'].std()  
ndf6['Outlier'] = abs(ndf6['Cargafamiliar'] - ndf6['Cargafamiliar'].mean()) > 1.96*ndf6['Cargafamiliar'].std()

"""
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Cargafamiliar'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Cargafamiliar'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Ingresomensualdesuresponsableeconomico"
for x in df['Ingresomensualdesuresponsableeconomico']:
   if RepresentsFloat(x) == False:
      z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
      valor = ""
      for y in z:
          valor = valor + y

   
      df.loc[df.Ingresomensualdesuresponsableeconomico == x, 'Ingresomensualdesuresponsableeconomico'] = valor


df.Ingresomensualdesuresponsableeconomico = df.Ingresomensualdesuresponsableeconomico.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Ingresomensualdesuresponsableeconomico'] - ndf6['Ingresomensualdesuresponsableeconomico'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Ingresomensualdesuresponsableeconomico'].std()  
ndf6['Outlier'] = abs(ndf6['Ingresomensualdesuresponsableeconomico'] - ndf6['Ingresomensualdesuresponsableeconomico'].mean()) > 1.96*ndf6['Ingresomensualdesuresponsableeconomico'].std()

    
""" 
#Grafico los outliers
idout = ndf6['ID'][(ndf6['Outlier'] == True)]
out = ndf6['Ingresomensualdesuresponsableeconomico'][(ndf6['Outlier'] == True)]
idnormal = ndf6['ID'][(ndf6['Outlier'] == False)]
normal = ndf6['Ingresomensualdesuresponsableeconomico'][(ndf6['Outlier'] == False)]
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
#------------------------------------------------------------------------------"Otrosingresos"
df['Otrosingresos'].fillna(0, inplace=True)
for x in df['Otrosingresos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Otrosingresos == x, 'Otrosingresos'] = valor
        else:
            
            df.loc[df.Otrosingresos == x, 'Otrosingresos'] = valor
           
         
df.Otrosingresos = df.Otrosingresos.astype(float)

#------------------------------------------------------------------------------"Totaldeingresos"
df['Totaldeingresos'] = df.Ingresomensualdesuresponsableeconomico + df.Otrosingresos 

#------------------------------------------------------------------------------"Gastosenviviendadesusresponsableseconomicos"
df['Gastosenviviendadesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosenviviendadesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosenviviendadesusresponsableseconomicos == x, 'Gastosenviviendadesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosenviviendadesusresponsableseconomicos == x, 'Gastosenviviendadesusresponsableseconomicos'] = valor
           
         
df.Gastosenviviendadesusresponsableseconomicos = df.Gastosenviviendadesusresponsableseconomicos.astype(float)

#------------------------------------------------------------------------------"Gastosenalimentaciondesusresponsableseconomicos"
df['Gastosenalimentaciondesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosenalimentaciondesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosenalimentaciondesusresponsableseconomicos == x, 'Gastosenalimentaciondesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosenalimentaciondesusresponsableseconomicos == x, 'Gastosenalimentaciondesusresponsableseconomicos'] = valor
           
         
df.Gastosenalimentaciondesusresponsableseconomicos = df.Gastosenalimentaciondesusresponsableseconomicos.astype(float)

#------------------------------------------------------------------------------"Gastosentransportedesusresponsableseconomicos"
df['Gastosentransportedesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosentransportedesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosentransportedesusresponsableseconomicos == x, 'Gastosentransportedesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosentransportedesusresponsableseconomicos == x, 'Gastosentransportedesusresponsableseconomicos'] = valor
           
         
df.Gastosentransportedesusresponsableseconomicos = df.Gastosentransportedesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Gastosmedicosdesusresponsableseconomicos"
df['Gastosmedicosdesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosmedicosdesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosmedicosdesusresponsableseconomicos == x, 'Gastosmedicosdesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosmedicosdesusresponsableseconomicos == x, 'Gastosmedicosdesusresponsableseconomicos'] = valor
           
         
df.Gastosmedicosdesusresponsableseconomicos = df.Gastosmedicosdesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Gastosodontologicosdesusresponsableseconomicos"
df['Gastosodontologicosdesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosodontologicosdesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosodontologicosdesusresponsableseconomicos == x, 'Gastosodontologicosdesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosodontologicosdesusresponsableseconomicos == x, 'Gastosodontologicosdesusresponsableseconomicos'] = valor
           
         
df.Gastosodontologicosdesusresponsableseconomicos = df.Gastosodontologicosdesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Gastoseducativosdesusresponsableseconomicos"
df['Gastoseducativosdesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastoseducativosdesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastoseducativosdesusresponsableseconomicos == x, 'Gastoseducativosdesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastoseducativosdesusresponsableseconomicos == x, 'Gastoseducativosdesusresponsableseconomicos'] = valor
           
         
df.Gastoseducativosdesusresponsableseconomicos = df.Gastoseducativosdesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos"
df['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos == x, 'Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos == x, 'Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'] = valor
           
         
df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos = df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Condominiodesusresponsableseconomicos"
df['Condominiodesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Condominiodesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Condominiodesusresponsableseconomicos == x, 'Condominiodesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Condominiodesusresponsableseconomicos == x, 'Condominiodesusresponsableseconomicos'] = valor
           
         
df.Condominiodesusresponsableseconomicos = df.Condominiodesusresponsableseconomicos.astype(float)
#------------------------------------------------------------------------------"Otrosgastosdesusresponsableseconomicos"
df['Otrosgastosdesusresponsableseconomicos'].fillna(0, inplace=True)
for x in df['Otrosgastosdesusresponsableseconomicos']:
    if RepresentsFloat(x) == False:
        
        z = re.split('[ \t,a-z]+', x, flags=re.IGNORECASE)
       
        valor = ""
        for y in z:
            valor = valor + y
        if RepresentsFloat(valor) == False:
                
            valor = "0"
            df.loc[df.Otrosgastosdesusresponsableseconomicos == x, 'Otrosgastosdesusresponsableseconomicos'] = valor
        else:
            
            df.loc[df.Otrosgastosdesusresponsableseconomicos == x, 'Otrosgastosdesusresponsableseconomicos'] = valor
           
         
df.Otrosgastosdesusresponsableseconomicos = df.Otrosgastosdesusresponsableseconomicos.astype(float)

#------------------------------------------------------------------------------"Totaldeegresosdesusresponsableseconomicos"
df['Totaldeegresosdesusresponsableseconomicos'] = df.Gastosenviviendadesusresponsableseconomicos + df.Gastosenalimentaciondesusresponsableseconomicos + df.Gastosentransportedesusresponsableseconomicos + df.Gastosmedicosdesusresponsableseconomicos + df.Gastosodontologicosdesusresponsableseconomicos + df.Gastoseducativosdesusresponsableseconomicos + df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos + df.Condominiodesusresponsableseconomicos + df.Otrosgastosdesusresponsableseconomicos
#------------------------------------------------------------------------------"DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE"
df.DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE = df.DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE.astype(int)
#------------------------------------------------------------------------------"Sugerenciasyrecomendacionesparamejorarnuestraatencion"

Ports = list(enumerate(np.unique(df['Sugerenciasyrecomendacionesparamejorarnuestraatencion'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Sugerenciasyrecomendacionesparamejorarnuestraatencion = df.Sugerenciasyrecomendacionesparamejorarnuestraatencion.map( lambda x: Ports_dict[x]).astype(int)    
#------------------------------------------------------------------------------
cols = ["ID", "SemestreAcademicoarenovar", "AnoAcademicoarenovar", "CedulaDeIdentidad", "DiadeNacimiento", "MesdeNacimiento", "AnodeNacimiento", "Edad", "EstadoCivil", "Sexo", "Escuela", "AnodeIngresoalaUCV", "ModalidaddeIngresoalaUCV", "Semestrequecursa", "Hacambiadousteddedireccion", "Deserafirmativoindiqueelmotivo", "Numerodemateriasinscritasenelsemestreoanoanterior", "Numerodemateriaaprobadasenelsemestreoanoanterior", "Numerodemateriasretiradasenelsemestreoanoanterior", "Numerodemateriasreprobadasenelsemestreoanoanterior", "Promedioponderadoaprobado", "Eficiencia", "Sireprobounaomasmateriasindiqueelmotivo", "Numerodemateriasinscritasenelsemestreencurso", "Seencuentrarealizandotesisopasantiasdegrado", "Cantidaddevecesqueharealizadotesisopasantiasdegrado", "Procedencia", "LugardonderesidemientrasestudiaenlaUniversidad", "Personasconlascualesustedvivemientrasestudiaenlauniversidad", "Tipodeviviendadonderesidemientrasestudiaenlauniversidad", "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual", "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada", "Contrajomatrimonio", "HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion", "Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo", "Seencuentraustedrealizandoalgunaactividadquelegenereingresos", "Encasodeserafirmativoindiquetipodeactividadysufrecuencia", "Montomensualdelabeca",  "Aportemensualquelebrindasuresponsableeconomico", "Aportemensualquerecibedefamiliaresoamigos", "Ingresomensualquerecibeporactividadesadestajooporhoras", "Ingresomensualtotal", "Gastosenalimentacionpersonal", "Gastosentransportepersonal", "Gastosmedicospersonal", "Gastosodontologicospersonal", "Gastospersonales", "Gastosenresidenciaohabitacionalquiladapersonal", "GastosenMaterialesdeestudiopersonal", "Gastosenrecreacionpersonal", "Otrosgastospersonal", "Totalegresospersonal", "Indiquequienessuresponsableeconomico", "Cargafamiliar", "Ingresomensualdesuresponsableeconomico", "Otrosingresos", "Totaldeingresos", "Gastosenviviendadesusresponsableseconomicos", "Gastosenalimentaciondesusresponsableseconomicos",  "Gastosentransportedesusresponsableseconomicos", "Gastosmedicosdesusresponsableseconomicos", "Gastosodontologicosdesusresponsableseconomicos", "Gastoseducativosdesusresponsableseconomicos", "Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos", "Condominiodesusresponsableseconomicos", "Otrosgastosdesusresponsableseconomicos", "Totaldeegresosdesusresponsableseconomicos", "DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE", "Sugerenciasyrecomendacionesparamejorarnuestraatencion"]
df = df[cols]
df = df.sort_values(by='ID', ascending= True)
df.describe()
#df.head()
#Genero el .cvs a partir del dataframe
df.to_csv("minable.csv")
