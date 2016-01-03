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
data2_file_object.writerow(["ID", "PeriodoAcademicoarenovar", "CedulaDeIdentidad", "Fecha.de.Nacimiento", "Edad", "Estado.Civil", "Sexo", "Escuela", "Año.de.Ingreso.a.la.UCV", "Modalidad.de.Ingreso.a.la.UCV", "Semestre que cursa", "Ha cambiado usted de dirección", "De.ser.afirmativo..indique.el.motivo", "Número.de.materias.inscritas.en.el.semestre.o.año.anterior", "Número.de.materias.aprobadas.en.el.semestre.o.año.anterior", "Número.de.materias.retiradas.en.el.semestre.o.año.anterior", "Número.de.materias.reprobadas.en.el.semestre.o.año.anterior", "Promedio.ponderado.aprobado", "Eficiencia", "Si.reprobó.una.o.más.materias.indique.el.motivo", "Número.de.materias.inscritas.en.el.semestre.en.curso", "Se.encuentra.realizando.tesis.o.pasantías.de.grado", "Cantidad.de.veces.que.ha.realizado.tesis.o.pasantías.de.grado", "Procedencia", "Lugar.donde.reside.mientras.estudia.en.el.Universidad", "Personas.con.las.cuales.usted.vive..mientras.estudia.en.la.universidad.", "Tipo.de.vivienda.donde.reside.mientras.estudia.en.la.universidad", "En.caso.de.vivir.en.habitación.alquilada.o.residencia.estudiantil..indique.el.monto.mensual.", "Dirección.donde.se.encuentra.ubicada.la.residencia.o.habitación.alquilada", "Contrajo.matrimonio.", "Ha.solicitado.algún.otro.beneficio.a.la.Universidad.u.otra.Institución.", "En.caso.afirmativo.señale.el.año.de.la.solicitud..institución.y.motivo", "Se.encuentra.usted..realizando.alguna.actividad.que.le.genere.ingresos.", "En.caso.de.ser.afirmativo..indique.tipo.de.actividad.y.su.frecuencia	", "Monto.mensual.de.la.beca",  "Aporte.mensual.que.le.brinda.su.responsable.económico", "Aporte.mensual.que.recibe.de.familiares.o.amigos", "Ingreso.mensual.que.recibe.por.actividades.a.destajo.o.por.horas", "Ingreso.mensual.total", "Gastos.en.alimentación.personal", "Gastos.en.transporte.personal", "Gastos.médicos.personal", "Gastos.odontológicos.personal", "Gastos.personales", "Gastos.en.residencia.o.habitación.alquilada.personal", "Gastos.en.Materiales.de.estudio.personal", "Gastos.en.recreación.personal", "Otros.gastos.personal", "Total.egresos.personal", "Indique.quién.es.su.responsable.económico", "Carga.familiar", "Ingreso.mensual.de.su.responsable.económico", "Otros.ingresos", "Total.de.ingresos", "Gastos.en.vivienda.de.sus.responsables.económicos ", "Gastos.en.alimentación.de.sus.responsables.económicos ",  "Gastos.en.transporte.de.sus.responsables.económicos ", "Gastos.médicos.de.sus.responsables.económicos ", "Gastos.odontológicos.de.sus.responsables.económicos ", "Gastos.educativos.de.sus.responsables.económicos", "Gastos.en.servicios.públicos.de.agua.luz.teléfono.y.gas.de.sus.responsables.económicos", "Condominio.de.sus.responsables.económicos", "Otros.gastos.de.sus.responsables.económicos", "Total.de.egresos.de.sus.responsables.económicos", "Deseamos.conocer.la.opinión.de.nuestros.usuarios..para.mejorar.la.calidad.de.los.servicios.ofrecidos.por.el.Dpto..de.Trabajo.Social.OBE", "Sugerencias.y.recomendaciones.para.mejorar.nuestra.atención"])
for row in data_file_object:       # For each row in test.csv
    data2_file_object.writerow(row)    # predict 0
data_file.close()
data2_file.close()
#-----------------------------------------------------------------
#Defino el DataFrame con todos los datos
df = pd.read_csv("data2.csv",header=0)

# All missing Embarked -> just make them embark from most common place
if len(df.PeriodoAcademicoarenovar[df.PeriodoAcademicoarenovar.isnull() ]) > 0:
    df.PeriodoAcademicoarenovar[ df.PeriodoAcademicoarenovar.isnull() ] = df.PeriodoAcademicoarenovar.dropna().mode().values

Ports = list(enumerate(np.unique(df['PeriodoAcademicoarenovar'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

#df.PeriodoAcademicoarenovar = df.PeriodoAcademicoarenovar.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
#------------------------------------------------------------------------------
#CONVIERTO TODA LA COLUMNA CEDULA EN FLOAT
df.CedulaDeIdentidad = df.CedulaDeIdentidad.astype(float)
#------------------------------------------------------------------------------
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
plt.show()
#------------------------------------------------------------------------------




#Genero el .cvs a partir del dataframe
#df.to_csv("minable.csv", sep='\t')
