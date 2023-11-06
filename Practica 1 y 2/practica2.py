import numpy as np
import matplotlib.pyplot as plt

# Load data from .dat files
# 600 seconds

t600_22Na = np.genfromtxt('Practica 1 y 2/NaI_22Na_600s.dat', names=True)
t600_60Co = np.genfromtxt('Practica 1 y 2/NaI_60Co_600s.dat', names=True)
t600_137Cs = np.genfromtxt('Practica 1 y 2/NaI_137Cs_600s.dat', names=True)
t600_fondo = np.genfromtxt('Practica 1 y 2/NaI_Fondo_600s.dat', names=True)
x_values_22Na = [x[0] for x in t600_22Na]
y_values_22Na = [x[1] for x in t600_22Na]
x_values_60Co = [x[0] for x in t600_60Co]
y_values_60Co = [x[1] for x in t600_60Co]
x_values_137Cs = [x[0] for x in t600_137Cs]
y_values_137Cs = [x[1] for x in t600_137Cs]
x_values_fondo = [x[0] for x in t600_fondo]
y_values_fondo = [x[1] for x in t600_fondo]

# 300 seconds
t300_57Co = np.genfromtxt('Practica 1 y 2/NaI_57Co_300s.dat', names=True)
t300_fondo = np.genfromtxt('Practica 1 y 2/NaI_Fondo_300s.dat', names=True)
x_values_57Co = [x[0] for x in t300_57Co]
y_values_57Co = [x[1] for x in t300_57Co]
x_values_fondo_300 = [x[0] for x in t300_fondo]
y_values_fondo_300 = [x[1] for x in t300_fondo]


## Preparación

#punto 1: graficar espectros
plt.figure(figsize=(12, 8))
plt.plot(x_values_22Na,y_values_22Na, label='22Na 600 s')
plt.plot(x_values_60Co,y_values_60Co, label='60Co 600 s')
plt.plot(x_values_137Cs,y_values_137Cs, label='137Cs 600 s')
plt.plot(x_values_fondo,y_values_fondo, label='Fondo 600 s')
plt.plot(x_values_57Co,y_values_57Co, label='57Co 300 s')
plt.plot(x_values_fondo_300,y_values_fondo_300, label='Fondo 300 s')
plt.title('Espectros de emisión para distintas fuentes y su fondo')
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('P1_espectros.png')
plt.show()

# Punto 2: Resta del fondo
y_values_22Na = [a - b for a, b in zip(y_values_22Na, y_values_fondo)]
y_values_60Co = [a - b for a, b in zip(y_values_60Co, y_values_fondo)]
y_values_137Cs = [a - b for a, b in zip(y_values_137Cs, y_values_fondo)]

y_values_57Co = [a - b for a, b in zip(y_values_57Co, y_values_fondo_300)]

# Punto 3: Espectro 57Co a 300s "Limpiado"
plt.figure(figsize=(12, 8))
plt.plot(x_values_57Co,y_values_57Co, label='57Co 300 s')
plt.title('Espectro de emisión para 57Co a 300s')
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('P3_57Co.png')
plt.show()

# Punto 4: Espectro 22Na a 600s "Limpiado"

plt.figure(figsize=(12, 8))
plt.plot(x_values_137Cs,y_values_137Cs, label='137Cs 600 s')
plt.title('Espectro de emisión para 137Cs a 600s')
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('P4_137Cs.png')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(x_values_22Na,y_values_22Na, label='22Na 600 s')
plt.plot(x_values_60Co,y_values_60Co, label='60Co 600 s')
plt.plot(x_values_137Cs,y_values_137Cs, label='137Cs 600 s')
plt.plot(x_values_57Co,y_values_57Co, label='57Co 300 s')
plt.title('Espectros de emisión para distintas fuentes sin fondo')
plt.xlim(right=500,left=0)
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('P4_Espectros.png')
plt.show()

# Punto 5: Calibración de picos

p_22Na_1 = 1274.537
p_60Co_1 = 1173.228
p_60Co_2 = 1332.490
p_137Cs_1 = 661.657
p_137Cs_alfa = 32.06
p_137Cs_beta = 36.66
p_57Co_1 = 122.06065
p_57Co_2 = 136.47350

# Punto 6: Encontrar el último máximo de la variable y_values_22Na, encontrar el valor correspondiente a x_values_22Na y graficar una linea en ese sitio que tenga un texto con el valor de la variable p_22Na_1

last_max_index = len(y_values_22Na) - y_values_22Na[::-1].index(max(y_values_22Na)) - 1
last_max_x = x_values_22Na[last_max_index]
plt.axvline(x=last_max_x, color='r')
plt.text(last_max_x, max(y_values_22Na), str(p_22Na_1), rotation=90)
plt.show()



