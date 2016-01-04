# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:37:56 2015
@author: eric
"""
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

# All missing Embarked -> just make them embark from most common place
if len(df.PeriodoAcademicoarenovar[df.PeriodoAcademicoarenovar.isnull() ]) > 0:
    df.PeriodoAcademicoarenovar[ df.PeriodoAcademicoarenovar.isnull() ] = df.PeriodoAcademicoarenovar.dropna().mode().values

Ports = list(enumerate(np.unique(df['PeriodoAcademicoarenovar'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

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
plt.figure()
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
plt.figure()

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
plt.figure()
#------------------------------------------------------------------------------Hacambiadousteddedireccion
df['Hacambiadousteddedireccion'] = df['Hacambiadousteddedireccion'].map( {'No': 0, 'Si': 1} ).astype(int)
#------------------------------------------------------------------------------Deserafirmativoindiqueelmotivo
df['Deserafirmativoindiqueelmotivo'].fillna('na', inplace=True)
print REVISAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAR
#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreoañoanterior
df.Numerodemateriasinscritasenelsemestreoanoanterior = df.Numerodemateriasinscritasenelsemestreoanoanterior.astype(float)

ndf2 = df.copy()
ndf2['x-Mean'] = abs(ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].mean())
ndf2['1.96*std'] = 1.96*ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].std()  
ndf2['Outlier'] = abs(ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'] - ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].mean()) > 1.96*ndf2['Numerodemateriasinscritasenelsemestreoanoanterior'].std()



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
plt.figure()
#------------------------------------------------------------------------------Numerodemateriaaprobadasenelsemestreoanoanterior
for x in df['Numerodemateriaaprobadasenelsemestreoanoanterior']:
   if RepresentsInt(x) == False:
      semestre = re.split('[a-z]+', x, flags=re.IGNORECASE)
      df.loc[df.Numerodemateriaaprobadasenelsemestreoanoanterior == x, 'Numerodemateriaaprobadasenelsemestreoanoanterior'] = semestre[0]
df.Numerodemateriaaprobadasenelsemestreoanoanterior = df.Numerodemateriaaprobadasenelsemestreoanoanterior.astype(float)
print ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR
#------------------------------------------------------------------------------Numerodemateriasretiradasenelsemestreoanoanterior
df.Numerodemateriasretiradasenelsemestreoanoanterior = df.Numerodemateriasretiradasenelsemestreoanoanterior.astype(float)
ndf4 = df.copy()
ndf4['x-Mean'] = abs(ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].mean())
ndf4['1.96*std'] = 1.96*ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].std()  
ndf4['Outlier'] = abs(ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'] - ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].mean()) > 1.96*ndf4['Numerodemateriasretiradasenelsemestreoanoanterior'].std()



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
plt.figure()
#------------------------------------------------------------------------------Numerodemateriasreprobadasenelsemestreoanoanterior
df.Numerodemateriasreprobadasenelsemestreoanoanterior = df.Numerodemateriasreprobadasenelsemestreoanoanterior.astype(float)
df.Numerodemateriasreprobadasenelsemestreoanoanterior 
ndf5 = df.copy()
ndf5['x-Mean'] = abs(ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean())
ndf5['1.96*std'] = 1.96*ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()  
ndf5['Outlier'] = abs(ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'] - ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].mean()) > 1.96*ndf5['Numerodemateriasreprobadasenelsemestreoanoanterior'].std()



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
plt.figure()

#------------------------------------------------------------------------------Promedioponderadoaprobado
df.Promedioponderadoaprobado = df.Promedioponderadoaprobado.astype(float)
for x in df['Promedioponderadoaprobado']:
    if x > 100000:
      valor = x/10000
      df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor

    if x >= 10000 and x < 100000:
        valor = x/1000
        
        df.loc[df.Promedioponderadoaprobado == x, 'Promedioponderadoaprobado'] = valor
    
 
   
df.Promedioponderadoaprobado[34]
#------------------------------------------------------------------------------Eficiencia
df.Eficiencia = df.Eficiencia.astype(float)
for x in df['Eficiencia']:
    if x > 10000 & x <= 10000:
        valor = x/100000
        print valor
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor
    if x > 1000 & x <= 10000
        valor = x/10000
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor
    if x > 100 & x <= 100:
        valor = x/1000
        df.loc[df.Eficiencia == x, 'Eficiencia'] = valor
       
df.Eficiencia      
ndf6 = df.copy()
ndf6['x-Mean'] = abs(ndf6['Eficiencia'] - ndf6['Eficiencia'].mean())
ndf6['1.96*std'] = 1.96*ndf6['Eficiencia'].std()  
ndf6['Outlier'] = abs(ndf6['Eficiencia'] - ndf6['Eficiencia'].mean()) > 1.96*ndf6['Eficiencia'].std()

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
plt.figure()

#------------------------------------------------------------------------------Sireprobounaomasmateriasindiqueelmotivo

#------------------------------------------------------------------------------Numerodemateriasinscritasenelsemestreencurso

#------------------------------------------------------------------------------Seencuentrarealizandotesisopasantiasdegrado

#------------------------------------------------------------------------------Cantidaddevecesqueharealizadotesisopasantiasdegrado

#------------------------------------------------------------------------------Procedencia

#------------------------------------------------------------------------------"LugardonderesidemientrasestudiaenlaUniversidad"
#------------------------------------------------------------------------------"Personasconlascualesustedvivemientrasestudiaenlauniversidad"
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
