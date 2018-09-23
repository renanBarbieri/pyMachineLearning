# -*- coding: utf-8 -*-
import numpy

##
# Trabalho de Aprendizado de máquina
# Inteligência Artificial - UFRJ - 2018.2
#
# Nome: Renan Hozumi Barbieri
# DRE: 111201610
# Nome: Felipe Gonçalves
# DRE:
#
##

# Pega os dados dos arquivos csv
# Para simplificação, adaptamos os dados para serem representados em números.
csv = numpy.genfromtxt ('./IrisDataset.csv', delimiter=",")
# Nos dados do arquivo IrisDataset.csv, realizamos a seguinte abstração
# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2

# csv = numpy.genfromtxt ('./DiscriminationInSalariesDataset.csv', delimiter=",")

dataLenght = len(csv)
x0 = numpy.ones(shape=(dataLenght, 1))
x = numpy.concatenate((x0,csv[:,:4]),axis=1)
y = csv[:,4]
w = numpy.array([0.1,0.2,0.3,0.4,0.5])

go = True
while go:
    go = False
    for i in range (dataLenght):
        h = numpy.dot(w.transpose() , x[i,:])

        if (numpy.sign(h) !=numpy.sign(y[i])):
            print ('Wrong classification.',numpy.sign(h), '!=', numpy.sign(y[i]))
            go = True
            print ('w before update:', w)
            w = w + numpy.dot(y[i],x[i,:])
            print('w updated:', w)
        else:
            print ('classificação correta.', numpy.sign(h), '=', numpy.sign(y[i]))
