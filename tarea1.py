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

pe2014 = df.loc[df['AnoAcademicoarenovar'] == 2014]
mode(pe2014['SemestreAcademicoarenovar'])

pe2015 = df.loc[df['AnoAcademicoarenovar'] == 2015]
mode(pe2015['SemestreAcademicoarenovar'])

x = mode(df['AnoAcademicoarenovar'] )
y = mode(df['SemestreAcademicoarenovar'] )

df['AnoAcademicoarenovar'].fillna(int(x[0]), inplace=True)
df['SemestreAcademicoarenovar'].fillna(int(y[0]), inplace=True)

df.AnoAcademicoarenovar = df.AnoAcademicoarenovar.astype(int)   
df.SemestreAcademicoarenovar = df.SemestreAcademicoarenovar.astype(int)  
#------------------------------------------------------------------------------CEDULA
#CONVIERTO TODA LA COLUMNA CEDULA EN FLOAT
df.CedulaDeIdentidad = df.CedulaDeIdentidad.astype(int)
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
      
#Hago una copia del dataframe, para hayar los OUTLIERS
newdf = df.copy()
newdf['x-Mean'] = abs(newdf['AnodeNacimiento'] - newdf['AnodeNacimiento'].mean())
newdf['1.96*std'] = 1.96*newdf['AnodeNacimiento'].std()  
newdf['Outlier'] = abs(newdf['AnodeNacimiento'] - newdf['AnodeNacimiento'].mean()) > 1.96*newdf['AnodeNacimiento'].std()

# data 
idout = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == True)]
out = newdf['AnodeNacimiento'][(newdf['Outlier'] == True)]
idnormal = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == False)]
normal = newdf['AnodeNacimiento'][(newdf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Ano de nacimiento')
plt.title('Outliers: Ano de nacimiento de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('AnoDeNacimiento.png')
         


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
idout = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == True)]
edadout = newdf['Edad'][(newdf['Outlier'] == True)]
idnormal = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == False)]
edadnormal = newdf['Edad'][(newdf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
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


ndf = df.copy()
ndf['x-Mean'] = abs(ndf['EstadoCivil'] - ndf['EstadoCivil'].mean())
ndf['1.96*std'] = 1.96*ndf['EstadoCivil'].std()  
ndf['Outlier'] = abs(ndf['EstadoCivil'] - ndf['EstadoCivil'].mean()) > 1.96*ndf['EstadoCivil'].std()


#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['EstadoCivil'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['EstadoCivil'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Estado civil del estudiante')
plt.title('Outliers: Estado civil de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('EstadoCivil.png')
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
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
AnodeIngresoalaUCVout = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
AnodeIngresoalaUCVnormal = ndf['AnodeIngresoalaUCV'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
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


ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Semestrequecursa'] - ndf['Semestrequecursa'].mean())
ndf['1.96*std'] = 1.96*ndf['Semestrequecursa'].std()  
ndf['Outlier'] = abs(ndf['Semestrequecursa'] - ndf['Semestrequecursa'].mean()) > 1.96*ndf['Semestrequecursa'].std()


#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Semestrequecursa'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Semestrequecursa'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Semestre que cursa el estudiante')
plt.title('Outliers: Semestre que cursan los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Semestrequecursa.png')

#------------------------------------------------------------------------------Hacambiadousteddedireccion
df['Hacambiadousteddedireccion'] = df['Hacambiadousteddedireccion'].map( {'No': 0, 'Si': 1} ).astype(int)
#mode(df['Hacambiadousteddedireccion'] )
#------------------------------------------------------------------------------Deserafirmativoindiqueelmotivo
df['Deserafirmativoindiqueelmotivo'].fillna('nan', inplace=True)

df['Deserafirmativoindiqueelmotivo'] = df['Deserafirmativoindiqueelmotivo'].map( {'nan': 0,'situacion de salud de un familiar': 1,'Unión a pareja': 2, 'porque mis padres se mudaron': 2,'No me llegan correos de ustedes': 3, 'mudanza': 2, 'mudanza ': 2,'vivia con mi tia, ahora vivo con mi madre en los valles del tuy': 2, 'olvido de clave': 4,'Por venta de la casa.': 2, 'Enfermedad de mi mamá': 1} ).astype(float)
df['Deserafirmativoindiqueelmotivo'] = df['Deserafirmativoindiqueelmotivo'].astype(int)

#respondieron = df.loc[df['Deserafirmativoindiqueelmotivo'] != 0]
#mode(respondieron['Deserafirmativoindiqueelmotivo'])
df['HacambiadousteddedireccionyMotivo'] = df['Hacambiadousteddedireccion']
#UNIFICACION
contador = 0
for x in df['HacambiadousteddedireccionyMotivo']:
     if x == 1:
         
         df['HacambiadousteddedireccionyMotivo'][contador] = df['Deserafirmativoindiqueelmotivo'][contador]
     contador= contador + 1


ndf = df.copy()
ndf['x-Mean'] = abs(ndf['HacambiadousteddedireccionyMotivo'] - ndf['HacambiadousteddedireccionyMotivo'].mean())
ndf['1.96*std'] = 1.96*ndf['HacambiadousteddedireccionyMotivo'].std()  
ndf['Outlier'] = abs(ndf['HacambiadousteddedireccionyMotivo'] - ndf['HacambiadousteddedireccionyMotivo'].mean()) > 1.96*ndf['HacambiadousteddedireccionyMotivo'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['HacambiadousteddedireccionyMotivo'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['HacambiadousteddedireccionyMotivo'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Ha cambiado de direccion el estudiante y el motivo')
plt.title('Outliers: Ha cambiado de direccion de los estudiantes y el motivo')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('HacambiadousteddedireccionyMotivo.png')
   
        
    

#------------------------------------------------------------------------------Numerodemateriaaprobadasenelsemestreoanoanterior
for x in df['Numerodemateriaaprobadasenelsemestreoanoanterior']:
   if RepresentsInt(x) == False:
      num = re.split('[a-z]+', x, flags=re.IGNORECASE)
      num
      df.loc[df.Numerodemateriaaprobadasenelsemestreoanoanterior == x, 'Numerodemateriaaprobadasenelsemestreoanoanterior'] = num[3]

df.Numerodemateriaaprobadasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior.astype(int)

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'] - ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'].mean())
ndf['1.96*std'] = 1.96*ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'].std()  
ndf['Outlier'] = abs(ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'] - ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'].mean()) > 1.96*ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Numerodemateriaaprobadasenelsemestreoanoanterior'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Numero de materias aprobadas en el semestre o ano anterior por el estudiante')
plt.title('Outliers: Numero de materias aprobadas en el semestre o ano anterior por los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Numerodemateriaaprobadasenelsemestreoanoanterior.png')


#------------------------------------------------------------------------------Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasretiradasenelsemestreoanoanterior = df.Numerodemateriasretiradasenelsemestreoanoanterior.astype(int)

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf['Numerodemateriasretiradasenelsemestreoanoanterior'].mean())
ndf['1.96*std'] = 1.96*ndf['Numerodemateriasretiradasenelsemestreoanoanterior'].std()  
ndf['Outlier'] = abs(ndf['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf['Numerodemateriasretiradasenelsemestreoanoanterior'].mean()) > 1.96*ndf['Numerodemateriasretiradasenelsemestreoanoanterior'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Numerodemateriasretiradasenelsemestreoanoanterior'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Numerodemateriasretiradasenelsemestreoanoanterior'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Numero de materias retiradas en el semestre o ano anterior por el estudiante')
plt.title('Outliers: Numero de materias retiradas en el semestre o ano anterior por los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Numerodemateriasretiradasenelsemestreoanoanterior.png')
#------------------------------------------------------------------------------Numerodemateriasreprobadasenelsemestreoanoanterior
df.Numerodemateriasreprobadasenelsemestreoanoanterior = df.Numerodemateriasreprobadasenelsemestreoanoanterior.astype(int)
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean())
ndf['1.96*std'] = 1.96*ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()  
ndf['Outlier'] = abs(ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean()) > 1.96*ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Numerodemateriasreprobadasenelsemestreoanoanterior'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Numero de materias reprobadas en el semestre o ano anterior por el estudiante')
plt.title('Outliers: Numero de materias reprobadas en el semestre o ano anterior por los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Numerodemateriasreprobadasenelsemestreoanoanterior.png')

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreoañoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior + df.Numerodemateriasreprobadasenelsemestreoanoanterior + df.Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriasinscritasenelsemestreoanoanterior.astype(int)

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf['Numerodemateriasinscritasenelsemestreoanoanterior'].mean())
ndf['1.96*std'] = 1.96*ndf['Numerodemateriasinscritasenelsemestreoanoanterior'].std()  
ndf['Outlier'] = abs(ndf['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf['Numerodemateriasinscritasenelsemestreoanoanterior'].mean()) > 1.96*ndf['Numerodemateriasinscritasenelsemestreoanoanterior'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Numerodemateriasinscritasenelsemestreoanoanterior'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Numerodemateriasinscritasenelsemestreoanoanterior'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Numero de materias inscritas en el semestre o ano anterior por el estudiante')
plt.title('Outliers: Numero de materias inscritas en el semestre o ano anterior por los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Numerodemateriasinscritasenelsemestreoanoanterior.png')
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
   
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Promedioponderadoaprobado'] - ndf['Promedioponderadoaprobado'].mean())
ndf['1.96*std'] = 1.96*ndf['Promedioponderadoaprobado'].std()  
ndf['Outlier'] = abs(ndf['Promedioponderadoaprobado'] - ndf['Promedioponderadoaprobado'].mean()) > 1.96*ndf['Promedioponderadoaprobado'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Promedioponderadoaprobado'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Promedioponderadoaprobado'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Promedio ponderado por el estudiante')
plt.title('Outliers: Promedio ponderado por los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Promedioponderadoaprobado.png')
#------------------------------------------------------------------------------Eficiencia
df['Eficiencia'].round(2)
for x in df['Eficiencia']:
    
    if int(x) >1:
        
        valor = "0." + str(int(x))
        
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor

df.Eficiencia = df.Eficiencia.astype(float)
df['Eficiencia'] = df['Eficiencia'].round(2) 
    
 
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Eficiencia'] - ndf['Eficiencia'].mean())
ndf['1.96*std'] = 1.96*ndf['Eficiencia'].std()  
ndf['Outlier'] = abs(ndf['Eficiencia'] - ndf['Eficiencia'].mean()) > 1.96*ndf['Eficiencia'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Eficiencia'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Eficiencia'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Eficiencia del estudiante')
plt.title('Outliers: Eficiencia de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Eficiencia.png')

#------------------------------------------------------------------------------Sireprobounaomasmateriasindiqueelmotivo

Ports = list(enumerate(np.unique(df['Sireprobounaomasmateriasindiqueelmotivo'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Sireprobounaomasmateriasindiqueelmotivo = df.Sireprobounaomasmateriasindiqueelmotivo.map( lambda x: Ports_dict[x]).astype(int)    

#mode(df['Sireprobounaomasmateriasindiqueelmotivo'])

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreencurso
df.Numerodemateriasinscritasenelsemestreencurso = df.Numerodemateriasinscritasenelsemestreencurso.astype(int)

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Numerodemateriasinscritasenelsemestreencurso'] - ndf['Numerodemateriasinscritasenelsemestreencurso'].mean())
ndf['1.96*std'] = 1.96*ndf['Numerodemateriasinscritasenelsemestreencurso'].std()  
ndf['Outlier'] = abs(ndf['Numerodemateriasinscritasenelsemestreencurso'] - ndf['Numerodemateriasinscritasenelsemestreencurso'].mean()) > 1.96*ndf['Numerodemateriasinscritasenelsemestreencurso'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Numerodemateriasinscritasenelsemestreencurso'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Numerodemateriasinscritasenelsemestreencurso'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Numero de materias inscritas en el semestre en curso del estudiante')
plt.title('Outliers: Numero de materias inscritas en el semestre en curso de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Numerodemateriasinscritasenelsemestreencurso.png')
#------------------------------------------------------------------------------Seencuentrarealizandotesisopasantiasdegrado
df['Seencuentrarealizandotesisopasantiasdegrado'] = df['Seencuentrarealizandotesisopasantiasdegrado'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------Cantidaddevecesqueharealizadotesisopasantiasdegrado

df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].fillna(0, inplace=True)
df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'] = df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].map( {0:0,'Primera vez': 1, 'Segunda vez': 2, 'Más de dos': 3} ).astype(float)
df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'] = df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'].astype(int)

df['SeencuentrarealizandotesisopasantiasdegradoyCantidad'] = df['Seencuentrarealizandotesisopasantiasdegrado']
#UNIFICACION
contador2 = 0
for x in df['SeencuentrarealizandotesisopasantiasdegradoyCantidad']:
    if x == 1:
        #print df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'][contador] 
        df['SeencuentrarealizandotesisopasantiasdegradoyCantidad'][contador2] = df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'][contador2]
        
    contador2 = contador2 + 1
        
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'] - ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'].mean())
ndf['1.96*std'] = 1.96*ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'].std()  
ndf['Outlier'] = abs(ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'] - ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'].mean()) > 1.96*ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['SeencuentrarealizandotesisopasantiasdegradoyCantidad'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Cantidad de veces que ha realizado tesis o pasantias de grado el estudiante')
plt.title('Outliers: Cantidad de veces que ha realizado tesis o pasantias de grado de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('SeencuentrarealizandotesisopasantiasdegradoyCantidad.png')

#------------------------------------------------------------------------------Procedencia
df['Procedencia'] = df['Procedencia'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25} ).astype(float)
df['Procedencia'] = df['Procedencia'].astype(int)

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Procedencia'] - ndf['Procedencia'].mean())
ndf['1.96*std'] = 1.96*ndf['Procedencia'].std()  
ndf['Outlier'] = abs(ndf['Procedencia'] - ndf['Procedencia'].mean()) > 1.96*ndf['Procedencia'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Procedencia'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Procedencia'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Procedencia del estudiante')
plt.title('Outliers: Procedencia de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Procedencia.png')
#------------------------------------------------------------------------------"LugardonderesidemientrasestudiaenlaUniversidad"
#SUstituyo los valores nulos por su procedencia
df['LugardonderesidemientrasestudiaenlaUniversidad'].fillna(0, inplace=True)
df['LugardonderesidemientrasestudiaenlaUniversidad'] = df['LugardonderesidemientrasestudiaenlaUniversidad'].map( {'Municipio Sucre':0,'Guarenas - Guatire': 1, 'Municipio Libertador Caracas': 2, 'Aragua': 3, 'Aragua ': 3,'Municipio Baruta': 4, 'Valles del Tuy': 5, 'Altos Mirandinos': 6,'Apure': 7, 'Municipio El Hatillo': 8, 'Municipio Chacao': 9,'Táchira': 10, 'Vargas':11, 'Monagas': 12, 'Portuguesa':13, 'Nueva Esparta': 14, 'Trujillo':15, 'Bolívar': 16, 'Barinas':17, 'Sucre': 18, 'Barlovento': 19, 'Anzoategui':20, 'Mérida': 21, 'Delta Amacuro': 22, 'Lara':23, 'Yaracuy': 24, 'Guárico': 25,0:26} ).astype(float)
contador=0

for x in df['LugardonderesidemientrasestudiaenlaUniversidad']:
    if x == 26:
        valor = df['Procedencia'][contador]    
        df['LugardonderesidemientrasestudiaenlaUniversidad'][contador] = valor
        
    contador=contador+1
    

df['LugardonderesidemientrasestudiaenlaUniversidad'] = df['LugardonderesidemientrasestudiaenlaUniversidad'].astype(int)   

ndf = df.copy()
ndf['x-Mean'] = abs(ndf['LugardonderesidemientrasestudiaenlaUniversidad'] - ndf['LugardonderesidemientrasestudiaenlaUniversidad'].mean())
ndf['1.96*std'] = 1.96*ndf['LugardonderesidemientrasestudiaenlaUniversidad'].std()  
ndf['Outlier'] = abs(ndf['LugardonderesidemientrasestudiaenlaUniversidad'] - ndf['LugardonderesidemientrasestudiaenlaUniversidad'].mean()) > 1.96*ndf['LugardonderesidemientrasestudiaenlaUniversidad'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['LugardonderesidemientrasestudiaenlaUniversidad'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['LugardonderesidemientrasestudiaenlaUniversidad'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Lugar donde reside mientras estudia en la Universidad el estudiante')
plt.title('Outliers: Lugar donde reside mientras estudia en la Universidad los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('LugardonderesidemientrasestudiaenlaUniversidad.png')

#------------------------------------------------------------------------------"Personasconlascualesustedvivemientrasestudiaenlauniversidad"
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] = df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].map( {'residencia estudiantil':0, 'recidencia':0, 'residencia':0,'Residencia':0,'Esposo (a) Hijos (as) ': 1,'Esposo (a) Hijos (as)\xc2\xa0': 1, 'Familiares paternos': 2, 'Madre': 3,'Familiares maternos': 4, 'Ambos padres': 5, 'Mi Mamá y mi hijo ': 6,'Padre': 7, 'Amigos': 8, 'madre y su esposo,abuela,y mi esposo': 9,'hermana': 10, 'Hermana': 10, 'dos hermanos':11, 'OTROS INQUILINOS': 12, 'hermanas':13, 'madre y hermana': 14, 'madre y hermanos':15, 'Madre y Hermanos':15, 'prima': 16, 'madrina':17, 'sola': 18, 'Mamá y Abuela': 19, 'madre,hermano e hijo':20, 'Madre, Hermano y Sobrina': 21, 'hermano, hermana y mi hijo': 22, 'compañeros de habitacion alquilada':23, 'Padres, hermana y abuelos maternos': 24, 'Madre, Hermana, Abuela': 25, 'abuela': 26, 'Dueños del apartamento donde alquilo la habitacion': 27, 'ambos padres y dos hermanis':28, 'Solo': 29, 'hermano':30, 'dueña del apartamento': 31, 'Madre y hermano': 32} ).astype(float)
df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] = df['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].astype(int) 
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] - ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].mean())
ndf['1.96*std'] = 1.96*ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].std()  
ndf['Outlier'] = abs(ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'] - ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].mean()) > 1.96*ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Personasconlascualesustedvivemientrasestudiaenlauniversidad'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Personas con las cuales vive el estudiante mientras estudian en la universidad')
plt.title('Outliers: Personas con las cuales viven los estudiantes mientras estudian en la universidad')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Personasconlascualesustedvivemientrasestudiaenlauniversidad.png')

#------------------------------------------------------------------------------"Tipodeviviendadonderesidemientrasestudiaenlauniversidad"
df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] = df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].map( {'Quinta o casa quinta':0, 'Apartamento en edifico': 1, 'Casa en barrio urbano': 2, 'Habitación alquilada': 3,'Casa de vecindad': 4, 'Residencia estudiantil': 5, 'Apartamento en quinta - casa quinta o casa': 6,'conserjería ': 7, 'Casa en barrio rural': 8, 'Casa de vecindad': 9,'casa': 10} ).astype(float)
df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] = df['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].astype(int) 
ndf = df.copy()
ndf['x-Mean'] = abs(ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] - ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].mean())
ndf['1.96*std'] = 1.96*ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].std()  
ndf['Outlier'] = abs(ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'] - ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].mean()) > 1.96*ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Tipodeviviendadonderesidemientrasestudiaenlauniversidad'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Tipo de vivienda donde reside el estudiante mientras estudia en la universidad')
plt.title('Outliers: Tipo de vivienda donde residen los estudiantes mientras estudian en la universidad')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Tipodeviviendadonderesidemientrasestudiaenlauniversidad.png')
#------------------------------------------------------------------------------ "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual"
#Los que tienen 0 significa que no viven en alquiler o residencia estudiantil o no respondieron
df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].fillna(0, inplace=True)
for x in df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual']:
    #Una unidad tributaria es equivalente a 125 bsf
    if x == "1 UT":
        df.loc[df.Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual == x, 'Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = "125"

    if RepresentsInt(x) == False:
        precio = re.split('[a-z]+', x, flags=re.IGNORECASE)
        df.loc[df.Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual == x, 'Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = precio[0]
df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] = df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].astype(float)

           
siviven = df.loc[df['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] != 0]
ndf = siviven.copy()
ndf['x-Mean'] = abs(ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] - ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].mean())
ndf['1.96*std'] = 1.96*ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].std()  
ndf['Outlier'] = abs(ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'] - ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].mean()) > 1.96*ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].std()

#Grafico los outliers
idout = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == True)]
valorout = ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'][(ndf['Outlier'] == True)]
idnormal = ndf['CedulaDeIdentidad'][(ndf['Outlier'] == False)]
valorUCVnormal = ndf['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'][(ndf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Monto mensual de la habitacion o residencial alquilada del estudiante')
plt.title('Outliers: Monto mensual de la habitacion o residencial alquilada de los estudiantes')
out = plt.scatter(idout, valorout, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, valorUCVnormal, marker='o', color=colors[1], s=100)
plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
#siviven['Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual'].describe()
#mode(respondieron['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'])
plt.savefig('Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual.png')

#------------------------------------------------------------------------------ "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada"

Ports = list(enumerate(np.unique(df['Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada = df.Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada.map( lambda x: Ports_dict[x]).astype(int)    
df['Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada'].fillna(0, inplace=True)


#------------------------------------------------------------------------------"Contrajomatrimonio"
df['Contrajomatrimonio'] = df['Contrajomatrimonio'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion."
df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion'] = df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo"
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].fillna(0, inplace=True)
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'] = df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].map( {0:0,'2012, OBE. AYUDA ECONÓMICA. SOLVENTAR GASTOS DE ESTUDIOS Y TRANSPORTE.':13, 'OBE 2011 UCV . Motivo: ayuda económica para mis estudios ': 1, 'AÑO 2011 AYUDA ECONÓMICA POR OBE PARA CUBRIR GASTOS  DE MATERIAL DE ESTUDIO.': 2, 'Beca ayudantía en el año 2011 Universidad Central de Venezuela (OBE) apoyo económico para material de estudio': 3,'solicitud en 2013. \nayuda economica \n': 4, 'Año solicitud 2013\nInstitución: Universidad Central de Venezuela (OBE)\nMotivo: Transporte, Materiales Estudios, Comida': 5, 'fames ayuda para maternidad': 6,'2015, ucv, falta de ingresos, ayuda economica': 7, 'Año: 2013. Institucion: OBE. Motivo: ayuda económica para rehabilitación (fisioterapia), ya que me atropello una moto.': 8, 'Año:2014 \nInstitucion: OBE\nMotivo: Compra de materiales de estudio. ': 9,'fecha:2014\ninstituto:OBE\ncompras de material de estudio': 10,'solicitud 2015 en el instituto OBE , motivo ayuda economica para gastos de la universidad (libro, pasaje, equipo de enfermeria, etc)': 11,'año 2014, OBE, solicitada para cubrir parte de los gastos mensuales estudiantiles': 12} ).astype(float)
df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'] = df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'].astype(int)



df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucionyMotivo'] = df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucion']
#UNIFICACION
contador3 = 0
for x in df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucionyMotivo']:
    if x == 1:
        #print df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'][contador] 
        df['HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucionyMotivo'][contador3] = df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'][contador3]
        
    contador3 = contador3 + 1
#respondieron = df.loc[df['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'] != 0]
#mode(respondieron['Encasoafirmativosenaleelanodelasolicitudinstitucionymotivo'])
#------------------------------------------------------------------------------ "Seencuentraustedrealizandoalgunaactividadquelegenereingresos"
df['Seencuentraustedrealizandoalgunaactividadquelegenereingresos'] = df['Seencuentraustedrealizandoalgunaactividadquelegenereingresos'].map( {'No': 0, 'Si': 1} ).astype(int)

#------------------------------------------------------------------------------"Encasodeserafirmativoindiquetipodeactividadysufrecuencia"
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].fillna(0, inplace=True)
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'] = df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].map( {0:0,'cuido pacientes parapoder obtener un ingreso economico': 1, 'STAFF DE EMPRESA DEPORTAIVA. UNO O DOS FINES DE SEMANA AL MES APROXIMADAMENTE.': 2, 'Recreación, algunos fines de semana (poco frecuente por falta de tiempo).': 3,'Secretaria, sólo los sábados medio turno.': 4} ).astype(float)
df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'] = df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'].astype(int)

respondieron = df.loc[df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'] != 0]
#mode(respondieron['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'])

df['SeencuentraustedrealizandoalgunaactividadquelegenereingresosIndiqueActividad'] = df['Seencuentraustedrealizandoalgunaactividadquelegenereingresos']
#UNIFICACION
contador4 = 0
for x in df['SeencuentraustedrealizandoalgunaactividadquelegenereingresosIndiqueActividad']:
    if x == 1:
        #print df['Cantidaddevecesqueharealizadotesisopasantiasdegrado'][contador] 
        df['SeencuentraustedrealizandoalgunaactividadquelegenereingresosIndiqueActividad'][contador4] = df['Encasodeserafirmativoindiquetipodeactividadysufrecuencia'][contador4]
        
    contador4 = contador4 + 1

#------------------------------------------------------------------------------"Montomensualdelabeca"
df.Montomensualdelabeca = df.Montomensualdelabeca.astype(float)
 
#Hago una copia del dataframe, para hayar los OUTLIERS
newdf = df.copy()
newdf['x-Mean'] = abs(newdf['Montomensualdelabeca'] - newdf['Montomensualdelabeca'].mean())
newdf['1.96*std'] = 1.96*newdf['Montomensualdelabeca'].std()  
newdf['Outlier'] = abs(newdf['Montomensualdelabeca'] - newdf['Montomensualdelabeca'].mean()) > 1.96*newdf['Montomensualdelabeca'].std()

# data 
idout = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == True)]
out = newdf['Montomensualdelabeca'][(newdf['Outlier'] == True)]
idnormal = newdf['CedulaDeIdentidad'][(newdf['Outlier'] == False)]
normal = newdf['Montomensualdelabeca'][(newdf['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Monto mensual de la beca del estudiante')
plt.title('Outliers: Monto mensual de la beca de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Montomensualdelabeca.png')
#------------------------------------------------------------------------------ "Aportemensualquelebrindasuresponsableeconomico"
df['Aportemensualquelebrindasuresponsableeconomico'].fillna(0, inplace=True)
df.Aportemensualquelebrindasuresponsableeconomico = df.Aportemensualquelebrindasuresponsableeconomico.astype(float)

#Hago una copia del dataframe, para hayar los OUTLIERS
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Aportemensualquelebrindasuresponsableeconomico'] - ndf6['Aportemensualquelebrindasuresponsableeconomico'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Aportemensualquelebrindasuresponsableeconomico'].std()  
ndf6['Outlier'] = abs(ndf6['Aportemensualquelebrindasuresponsableeconomico'] - ndf6['Aportemensualquelebrindasuresponsableeconomico'].mean()) > 1.96*ndf6['Aportemensualquelebrindasuresponsableeconomico'].std()

# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Aportemensualquelebrindasuresponsableeconomico'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Aportemensualquelebrindasuresponsableeconomico'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Aporte mensual que le brinda el responsable economico del estudiante')
plt.title('Outliers: Aporte mensual que le brinda el responsable economico de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Aportemensualquelebrindasuresponsableeconomico.png')
#------------------------------------------------------------------------------"Aportemensualquerecibedefamiliaresoamigos"
df['Aportemensualquerecibedefamiliaresoamigos'].fillna(0, inplace=True)
df.Aportemensualquerecibedefamiliaresoamigos = df.Aportemensualquerecibedefamiliaresoamigos.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Aportemensualquerecibedefamiliaresoamigos'] - ndf6['Aportemensualquerecibedefamiliaresoamigos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Aportemensualquerecibedefamiliaresoamigos'].std()  
ndf6['Outlier'] = abs(ndf6['Aportemensualquerecibedefamiliaresoamigos'] - ndf6['Aportemensualquerecibedefamiliaresoamigos'].mean()) > 1.96*ndf6['Aportemensualquerecibedefamiliaresoamigos'].std()
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Aportemensualquerecibedefamiliaresoamigos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Aportemensualquerecibedefamiliaresoamigos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Aporte mensual que recibe de familiares o amigos el estudiante')
plt.title('Outliers: Aporte mensual que recibe de familiares o amigos los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Aportemensualquerecibedefamiliaresoamigos.png')

#------------------------------------------------------------------------------"Ingresomensualquerecibeporactividadesadestajooporhoras"

df['Ingresomensualquerecibeporactividadesadestajooporhoras'].fillna(0, inplace=True)
df.Ingresomensualquerecibeporactividadesadestajooporhoras = df.Ingresomensualquerecibeporactividadesadestajooporhoras.astype(float)


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'] - ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].std()  
ndf6['Outlier'] = abs(ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'] - ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].mean()) > 1.96*ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'].std()


#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Ingresomensualquerecibeporactividadesadestajooporhoras'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Ingreso mensual que recibe por actividades de trabajo por horas  el estudiante')
plt.title('Outliers: Ingreso mensual que reciben por actividades de trabajo por horas los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Ingresomensualquerecibeporactividadesadestajooporhoras.png')
#------------------------------------------------------------------------------"Ingresomensualtotal"
df['Ingresomensualtotal'] = df.Montomensualdelabeca + df.Aportemensualquelebrindasuresponsableeconomico + df.Aportemensualquerecibedefamiliaresoamigos + df.Ingresomensualquerecibeporactividadesadestajooporhoras
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Ingresomensualtotal'] - ndf6['Ingresomensualtotal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Ingresomensualtotal'].std()  
ndf6['Outlier'] = abs(ndf6['Ingresomensualtotal'] - ndf6['Ingresomensualtotal'].mean()) > 1.96*ndf6['Ingresomensualtotal'].std()


#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Ingresomensualtotal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Ingresomensualtotal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Ingreso mensual total del estudiante')
plt.title('Outliers: Ingreso mensual total de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Ingresomensualtotal.png')


#------------------------------------------------------------------------------"Gastosenalimentacionpersonal"
df['Gastosenalimentacionpersonal'].fillna(0, inplace=True)
df.Gastosenalimentacionpersonal = df.Gastosenalimentacionpersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenalimentacionpersonal'] - ndf6['Gastosenalimentacionpersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenalimentacionpersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenalimentacionpersonal'] - ndf6['Gastosenalimentacionpersonal'].mean()) > 1.96*ndf6['Gastosenalimentacionpersonal'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenalimentacionpersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenalimentacionpersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en alimentacion personal del estudiante')
plt.title('Outliers: Gastos en alimentacion personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenalimentacionpersonal.png')
#------------------------------------------------------------------------------"Gastosentransportepersonal"
df['Gastosentransportepersonal'].fillna(0, inplace=True)
df.Gastosentransportepersonal = df.Gastosentransportepersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosentransportepersonal'] - ndf6['Gastosentransportepersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosentransportepersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosentransportepersonal'] - ndf6['Gastosentransportepersonal'].mean()) > 1.96*ndf6['Gastosentransportepersonal'].std()
#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosentransportepersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosentransportepersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en transporte personal del estudiante')
plt.title('Outliers: Gastos en transporte personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosentransportepersonal.png')
#------------------------------------------------------------------------------"Gastosmedicospersonal"
df['Gastosmedicospersonal'].fillna(0, inplace=True)
df.Gastosmedicospersonal = df.Gastosmedicospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosmedicospersonal'] - ndf6['Gastosmedicospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosmedicospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosmedicospersonal'] - ndf6['Gastosmedicospersonal'].mean()) > 1.96*ndf6['Gastosmedicospersonal'].std()
#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosmedicospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosmedicospersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos medicos personal del estudiante')
plt.title('Outliers: Gastos medicos personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosmedicospersonal.png')
#------------------------------------------------------------------------------"Gastosodontologicospersonal"
df['Gastosodontologicospersonal'].fillna(0, inplace=True)
df.Gastosodontologicospersonal = df.Gastosodontologicospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosodontologicospersonal'] - ndf6['Gastosodontologicospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosodontologicospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosodontologicospersonal'] - ndf6['Gastosodontologicospersonal'].mean()) > 1.96*ndf6['Gastosodontologicospersonal'].std()
#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosodontologicospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosodontologicospersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos odontologicos personal del estudiante')
plt.title('Outliers: Gastos odontologicos personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosodontologicospersonal.png')
#------------------------------------------------------------------------------"Gastospersonales"
df['Gastospersonales'].fillna(0, inplace=True)
df.Gastospersonales = df.Gastospersonales.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastospersonales'] - ndf6['Gastospersonales'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastospersonales'].std()  
ndf6['Outlier'] = abs(ndf6['Gastospersonales'] - ndf6['Gastospersonales'].mean()) > 1.96*ndf6['Gastospersonales'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastospersonales'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastospersonales'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos personales del estudiante')
plt.title('Outliers: Gastos personales de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastospersonales.png')
#------------------------------------------------------------------------------"Gastosenresidenciaohabitacionalquiladapersonal"
df['Gastosenresidenciaohabitacionalquiladapersonal'].fillna(0, inplace=True)
df.Gastosenresidenciaohabitacionalquiladapersonal = df.Gastosenresidenciaohabitacionalquiladapersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenresidenciaohabitacionalquiladapersonal'] - ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenresidenciaohabitacionalquiladapersonal'] - ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].mean()) > 1.96*ndf6['Gastosenresidenciaohabitacionalquiladapersonal'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenresidenciaohabitacionalquiladapersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenresidenciaohabitacionalquiladapersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en residencia o habitacion alquilada personal del estudiante')
plt.title('Outliers: Gastos en residencia o habitacion alquilada personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenresidenciaohabitacionalquiladapersonal.png')
#------------------------------------------------------------------------------"GastosenMaterialesdeestudiopersonal"
df['GastosenMaterialesdeestudiopersonal'].fillna(0, inplace=True)
df.GastosenMaterialesdeestudiopersonal = df.GastosenMaterialesdeestudiopersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['GastosenMaterialesdeestudiopersonal'] - ndf6['GastosenMaterialesdeestudiopersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['GastosenMaterialesdeestudiopersonal'].std()  
ndf6['Outlier'] = abs(ndf6['GastosenMaterialesdeestudiopersonal'] - ndf6['GastosenMaterialesdeestudiopersonal'].mean()) > 1.96*ndf6['GastosenMaterialesdeestudiopersonal'].std()


#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['GastosenMaterialesdeestudiopersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['GastosenMaterialesdeestudiopersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en materiales de estudio personal del estudiante')
plt.title('Outliers: Gastos en materiales de estudio personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('GastosenMaterialesdeestudiopersonal.png')
#------------------------------------------------------------------------------"Gastosenrecreacionpersonal"
df['Gastosenrecreacionpersonal'].fillna(0, inplace=True)
df.Gastosenrecreacionpersonal = df.Gastosenrecreacionpersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenrecreacionpersonal'] - ndf6['Gastosenrecreacionpersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenrecreacionpersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenrecreacionpersonal'] - ndf6['Gastosenrecreacionpersonal'].mean()) > 1.96*ndf6['Gastosenrecreacionpersonal'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenrecreacionpersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenrecreacionpersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en recreacion personal del estudiante')
plt.title('Outliers: Gastos en recreacion personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenrecreacionpersonal.png')
#------------------------------------------------------------------------------"Otrosgastospersonal"
df['Otrosgastospersonal'].fillna(0, inplace=True)
df.Otrosgastospersonal = df.Otrosgastospersonal.astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Otrosgastospersonal'] - ndf6['Otrosgastospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Otrosgastospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Otrosgastospersonal'] - ndf6['Otrosgastospersonal'].mean()) > 1.96*ndf6['Otrosgastospersonal'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Otrosgastospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Otrosgastospersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Otros gastos personales del estudiante')
plt.title('Outliers: Otros gastos personales de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Otrosgastospersonal.png')

#------------------------------------------------------------------------------"Totalegresospersonal"
df['Totalegresospersonal'] = df.Gastosenalimentacionpersonal + df.Gastosentransportepersonal + df.Gastosmedicospersonal + df.Gastosodontologicospersonal + df.Gastospersonales + df.Gastosenresidenciaohabitacionalquiladapersonal + df.GastosenMaterialesdeestudiopersonal + df.Gastosenrecreacionpersonal + df.Otrosgastospersonal
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Totalegresospersonal'] - ndf6['Totalegresospersonal'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Totalegresospersonal'].std()  
ndf6['Outlier'] = abs(ndf6['Totalegresospersonal'] - ndf6['Totalegresospersonal'].mean()) > 1.96*ndf6['Totalegresospersonal'].std()


#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Totalegresospersonal'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Totalegresospersonal'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Total egresos personal del estudiante')
plt.title('Outliers: Total egresos personal de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Totalegresospersonal.png')

#------------------------------------------------------------------------------"Indiquequienessuresponsableeconomico"
df['Indiquequienessuresponsableeconomico'] = df['Indiquequienessuresponsableeconomico'].map( {'Usted mismo':0,'esposo': 1, 'ninguno': 2, 'Madre': 3,'Familiares': 4, 'Padre': 5, 'Ambos padres': 6,'Cónyugue': 7, 'tia': 8, 'Hermano': 9,'hermana': 10,'MI HERMANA': 10, 'abuela': 10, 'dos hermanos':11, 'OTROS INQUILINOS': 12, 'hermanas':13, 'madre y hermana': 14, 'madre y hermanos':15, 'Madre y Hermanos':15, 'prima': 16, 'madrina':17, 'sola': 18, 'Mamá y Abuela': 19, 'madre,hermano e hijo':20, 'Madre, Hermano y Sobrina': 21, 'hermano, hermana y mi hijo': 22, 'compañeros de habitacion alquilada':23, 'Padres, hermana y abuelos maternos': 24, 'Madre, Hermana, Abuela': 25, 'abuela': 26, 'Dueños del apartamento donde alquilo la habitacion': 27, 'ambos padres y dos hermanis':28, 'Solo': 29, 'hermano':30, 'dueña del apartamento': 31, 'Madre y hermano': 32} ).astype(float)
df['Indiquequienessuresponsableeconomico'] = df['Indiquequienessuresponsableeconomico'].astype(int)

#------------------------------------------------------------------------------"Cargafamiliar"
df.Cargafamiliar = df.Cargafamiliar.astype(int)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Cargafamiliar'] - ndf6['Cargafamiliar'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Cargafamiliar'].std()  
ndf6['Outlier'] = abs(ndf6['Cargafamiliar'] - ndf6['Cargafamiliar'].mean()) > 1.96*ndf6['Cargafamiliar'].std()


#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Cargafamiliar'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Cargafamiliar'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Carga familiar del estudiante')
plt.title('Outliers: Carga familiar de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Cargafamiliar.png')
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

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Ingresomensualdesuresponsableeconomico'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Ingresomensualdesuresponsableeconomico'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Ingreso mensual del responsable economico del estudiante')
plt.title('Outliers: Ingreso mensual del responsable economico de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Ingresomensualdesuresponsableeconomico.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Otrosingresos'] - ndf6['Otrosingresos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Otrosingresos'].std()  
ndf6['Outlier'] = abs(ndf6['Otrosingresos'] - ndf6['Otrosingresos'].mean()) > 1.96*ndf6['Otrosingresos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Otrosingresos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Otrosingresos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Otros ingresos del estudiante')
plt.title('Outliers: Otros ingresos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Otrosingresos.png')

#------------------------------------------------------------------------------"Totaldeingresos"
df['Totaldeingresos'] = df.Ingresomensualdesuresponsableeconomico + df.Otrosingresos 
df['Totaldeingresos'] = df['Totaldeingresos'].astype(float)

#df_norm = (df['Totaldeingresos'] - df['Totaldeingresos'].mean()) / (df['Totaldeingresos'].max() - df['Totaldeingresos'].min())

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Totaldeingresos'] - ndf6['Totaldeingresos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Totaldeingresos'].std()  
ndf6['Outlier'] = abs(ndf6['Totaldeingresos'] - ndf6['Totaldeingresos'].mean()) > 1.96*ndf6['Totaldeingresos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Totaldeingresos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Totaldeingresos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Total de ingresos del estudiante')
plt.title('Outliers: Total de ingresos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Totaldeingresos.png')
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


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenviviendadesusresponsableseconomicos'] - ndf6['Gastosenviviendadesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenviviendadesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenviviendadesusresponsableseconomicos'] - ndf6['Gastosenviviendadesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosenviviendadesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenviviendadesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenviviendadesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en vivienda de los responsables economicos del estudiante')
plt.title('Outliers: Gastos en vivienda de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenviviendadesusresponsableseconomicos.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenalimentaciondesusresponsableseconomicos'] - ndf6['Gastosenalimentaciondesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenalimentaciondesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenalimentaciondesusresponsableseconomicos'] - ndf6['Gastosenalimentaciondesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosenalimentaciondesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenalimentaciondesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenalimentaciondesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en alimentacion de los responsables economicos del estudiante')
plt.title('Outliers: Gastos en alimentacion de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenalimentaciondesusresponsableseconomicos.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosentransportedesusresponsableseconomicos'] - ndf6['Gastosentransportedesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosentransportedesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosentransportedesusresponsableseconomicos'] - ndf6['Gastosentransportedesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosentransportedesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosentransportedesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosentransportedesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en transporte de los responsables economicos del estudiante')
plt.title('Outliers: Gastos en transporte de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosentransportedesusresponsableseconomicos.png')
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


ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosmedicosdesusresponsableseconomicos'] - ndf6['Gastosmedicosdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosmedicosdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosmedicosdesusresponsableseconomicos'] - ndf6['Gastosmedicosdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosmedicosdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosmedicosdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosmedicosdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos medicos de los responsables economicos del estudiante')
plt.title('Outliers: Gastos medicos de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosmedicosdesusresponsableseconomicos.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosodontologicosdesusresponsableseconomicos'] - ndf6['Gastosodontologicosdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosodontologicosdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosodontologicosdesusresponsableseconomicos'] - ndf6['Gastosodontologicosdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosodontologicosdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosodontologicosdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosodontologicosdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos odontologicos de los responsables economicos del estudiante')
plt.title('Outliers: Gastos odontologicos de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosodontologicosdesusresponsableseconomicos.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastoseducativosdesusresponsableseconomicos'] - ndf6['Gastoseducativosdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastoseducativosdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastoseducativosdesusresponsableseconomicos'] - ndf6['Gastoseducativosdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastoseducativosdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastoseducativosdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastoseducativosdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos educativos de los responsables economicos del estudiante')
plt.title('Outliers: Gastos educativos de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastoseducativosdesusresponsableseconomicos.png')

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
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'] - ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'] - ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en servicios publicos de agua luz telefono y gas de los responsables economicos del estudiante')
plt.title('Outliers: Gastos en servicios publicos de agua luz telefono y gas de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos.png')

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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Condominiodesusresponsableseconomicos'] - ndf6['Condominiodesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Condominiodesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Condominiodesusresponsableseconomicos'] - ndf6['Condominiodesusresponsableseconomicos'].mean()) > 1.96*ndf6['Condominiodesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Condominiodesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Condominiodesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Gastos en condominio de los responsables economicos del estudiante')
plt.title('Outliers: Gastos en condominio de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Condominiodesusresponsableseconomicos.png')
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

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Otrosgastosdesusresponsableseconomicos'] - ndf6['Otrosgastosdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Otrosgastosdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Otrosgastosdesusresponsableseconomicos'] - ndf6['Otrosgastosdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Otrosgastosdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Otrosgastosdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Otrosgastosdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Otros gastos responsables economicos del estudiante')
plt.title('Outliers: Otros gastos de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Otrosgastosdesusresponsableseconomicos.png')

#------------------------------------------------------------------------------"Totaldeegresosdesusresponsableseconomicos"
df['Totaldeegresosdesusresponsableseconomicos'] = df.Gastosenviviendadesusresponsableseconomicos + df.Gastosenalimentaciondesusresponsableseconomicos + df.Gastosentransportedesusresponsableseconomicos + df.Gastosmedicosdesusresponsableseconomicos + df.Gastosodontologicosdesusresponsableseconomicos + df.Gastoseducativosdesusresponsableseconomicos + df.Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos + df.Condominiodesusresponsableseconomicos + df.Otrosgastosdesusresponsableseconomicos
df['Totaldeegresosdesusresponsableseconomicos'] = df['Totaldeegresosdesusresponsableseconomicos'].astype(float)

ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Totaldeegresosdesusresponsableseconomicos'] - ndf6['Totaldeegresosdesusresponsableseconomicos'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Totaldeegresosdesusresponsableseconomicos'].std()  
ndf6['Outlier'] = abs(ndf6['Totaldeegresosdesusresponsableseconomicos'] - ndf6['Totaldeegresosdesusresponsableseconomicos'].mean()) > 1.96*ndf6['Totaldeegresosdesusresponsableseconomicos'].std()

#Grafico los outliers
# data 
idout = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == True)]
out = ndf6['Totaldeegresosdesusresponsableseconomicos'][(ndf6['Outlier'] == True)]
idnormal = ndf6['CedulaDeIdentidad'][(ndf6['Outlier'] == False)]
normal = ndf6['Totaldeegresosdesusresponsableseconomicos'][(ndf6['Outlier'] == False)]
colors = ['r', 'b']
plt.figure()
plt.xlabel('Cedula del estudiante')
plt.ylabel('Total de egresos de los responsables economicos del estudiante')
plt.title('Outliers: Total de egresos de los responsables economicos de los estudiantes')
out = plt.scatter(idout, out, marker='x', color=colors[0], s=100)
rest = plt.scatter(idnormal, normal, marker='o', color=colors[1], s=100)

plt.legend((out, rest),
           ('Outlier', 'Estudiante promedio'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
           

plt.savefig('Totaldeegresosdesusresponsableseconomicos.png')

#------------------------------------------------------------------------------"DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE"
df.DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE = df.DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE.astype(int)
mode(df.DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE)
#------------------------------------------------------------------------------"Sugerenciasyrecomendacionesparamejorarnuestraatencion"

Ports = list(enumerate(np.unique(df['Sugerenciasyrecomendacionesparamejorarnuestraatencion'])))   
Ports_dict = { name : i for i, name in Ports }              

df.Sugerenciasyrecomendacionesparamejorarnuestraatencion = df.Sugerenciasyrecomendacionesparamejorarnuestraatencion.map( lambda x: Ports_dict[x]).astype(int)    
mode(df.Sugerenciasyrecomendacionesparamejorarnuestraatencion)
#------------------------------------------------------------------------------
cols = ["CedulaDeIdentidad", "SemestreAcademicoarenovar", "AnoAcademicoarenovar", "DiadeNacimiento", "MesdeNacimiento", "AnodeNacimiento", "EstadoCivil", "Sexo", "Escuela", "AnodeIngresoalaUCV", "ModalidaddeIngresoalaUCV", "Semestrequecursa", 'HacambiadousteddedireccionyMotivo', "Numerodemateriaaprobadasenelsemestreoanoanterior", "Numerodemateriasretiradasenelsemestreoanoanterior", "Numerodemateriasreprobadasenelsemestreoanoanterior", "Promedioponderadoaprobado", "Eficiencia", "Sireprobounaomasmateriasindiqueelmotivo", "Numerodemateriasinscritasenelsemestreencurso", "SeencuentrarealizandotesisopasantiasdegradoyCantidad", "Procedencia", "LugardonderesidemientrasestudiaenlaUniversidad", "Personasconlascualesustedvivemientrasestudiaenlauniversidad", "Tipodeviviendadonderesidemientrasestudiaenlauniversidad", "Encasodevivirenhabitacionalquiladaoresidenciaestudiantilindiqueelmontomensual", "Direcciondondeseencuentraubicadalaresidenciaohabitacionalquilada", "HasolicitadoalgunotrobeneficioalaUniversidaduotraInstitucionyMotivo", "SeencuentraustedrealizandoalgunaactividadquelegenereingresosIndiqueActividad", "Montomensualdelabeca",  "Aportemensualquelebrindasuresponsableeconomico", "Aportemensualquerecibedefamiliaresoamigos", "Ingresomensualquerecibeporactividadesadestajooporhoras", "Gastosenalimentacionpersonal", "Gastosentransportepersonal", "Gastosmedicospersonal", "Gastosodontologicospersonal", "Gastospersonales", "Gastosenresidenciaohabitacionalquiladapersonal", "GastosenMaterialesdeestudiopersonal", "Gastosenrecreacionpersonal", "Otrosgastospersonal", "Indiquequienessuresponsableeconomico", "Cargafamiliar", "Ingresomensualdesuresponsableeconomico", "Otrosingresos", "Gastosenviviendadesusresponsableseconomicos", "Gastosenalimentaciondesusresponsableseconomicos",  "Gastosentransportedesusresponsableseconomicos", "Gastosmedicosdesusresponsableseconomicos", "Gastosodontologicosdesusresponsableseconomicos", "Gastoseducativosdesusresponsableseconomicos", "Gastosenserviciospublicosdeagualuztelefonoygasdesusresponsableseconomicos", "Condominiodesusresponsableseconomicos", "Otrosgastosdesusresponsableseconomicos", "DeseamosconocerlaopiniondenuestrosusuariosparamejorarlacalidaddelosserviciosofrecidosporelDptodeTrabajoSocialOBE", "Sugerenciasyrecomendacionesparamejorarnuestraatencion"]
df = df[cols]
df = df.sort_values(by='CedulaDeIdentidad', ascending= True)
df.info()
#df.describe()
#df.head()
#Genero el .cvs a partir del dataframe
df.to_csv("minable.csv")
