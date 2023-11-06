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

max_i_22Na = len(y_values_22Na[400:]) - y_values_22Na[400:][::-1].index(max(y_values_22Na[400:])) - 1 + 400
max_x_22Na = x_values_22Na[max_i_22Na]
# Para los índices 300-400
max_i_60Co_1 = y_values_60Co[300:400].index(max(y_values_60Co[300:400])) + 300
max_x_60Co_1 = x_values_60Co[max_i_60Co_1]
# Para los índices 400 en adelante
max_i_60Co_2 = y_values_60Co[400:].index(max(y_values_60Co[400:])) + 400
max_x_60Co_2 = x_values_60Co[max_i_60Co_2]

max_i_137Cs_1 = y_values_137Cs[:100].index(max(y_values_137Cs[:100]))
max_x_137Cs_1 = x_values_137Cs[max_i_137Cs_1]

max_i_137Cs_2 = y_values_137Cs[200:300].index(max(y_values_137Cs[200:300])) + 200
max_x_137Cs_2 = x_values_137Cs[max_i_137Cs_2]

max_i_57Co_1 = y_values_57Co[:100].index(max(y_values_57Co[:100]))
max_x_57Co_1 = x_values_57Co[max_i_57Co_1]

plt.figure(figsize=(12, 8))

plt.axvline(x=max_x_22Na, color='r')
plt.axvline(x=max_x_60Co_1, color='r')
plt.axvline(x=max_x_60Co_2, color='r')
plt.axvline(x=max_x_137Cs_1, color='r')
plt.axvline(x=max_x_137Cs_2, color='r')
plt.axvline(x=max_x_57Co_1, color='r')

plt.plot(x_values_22Na,y_values_22Na, label='22Na 600 s')
plt.plot(x_values_60Co, y_values_60Co, label='60Co 600 s')
plt.plot(x_values_137Cs, y_values_137Cs, label='137Cs 600 s')
plt.plot(x_values_57Co, y_values_57Co, label='57Co 300 s')

plt.text(max_x_22Na, max(y_values_22Na), "{:.1f} keV".format(p_22Na_1), rotation=90)
plt.text(max_x_60Co_1, max(y_values_60Co[300:400]), "{:.1f} keV".format(p_60Co_1), rotation=90)
plt.text(max_x_60Co_2, max(y_values_60Co[400:]), "{:.1f} keV".format(p_60Co_2), rotation=90)
plt.text(max_x_137Cs_1, max(y_values_137Cs[:100]), "{:.1f} keV".format(p_137Cs_alfa), rotation=90)
plt.text(max_x_137Cs_1, max(y_values_137Cs[200:300])-2000, "{:.1f} keV".format(p_137Cs_beta), rotation=90)
plt.text(max_x_137Cs_2, max(y_values_137Cs[200:300]), "{:.1f} keV".format(p_137Cs_1), rotation=90)
plt.text(max_x_57Co_1, max(y_values_57Co[:100]), "{:.1f} keV".format(p_57Co_1), rotation=90)
plt.text(max_x_57Co_1, max(y_values_57Co[:100])+2500, "{:.1f} keV".format(p_57Co_2), rotation=90)

plt.xlim(right=500,left=0)
plt.ylim(top=14000)

plt.title('Espectros de emisión con picos de energía asociados')

plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('P5_picos_energia.png')
plt.show()

# Punto 6: Ajustes de gaussiana
# ANALISIS

# Punto 1: Ajuste de gaussianas
# Punto 1.1: Ajuste de gaussiana para 22Na
from scipy.optimize import curve_fit

def gaussiana(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

x_values_22Na = np.array(x_values_22Na)
y_values_22Na = np.array(y_values_22Na)

popt_22Na, pcov_22Na = curve_fit(gaussiana, x_values_22Na, y_values_22Na, p0=[max(y_values_22Na), max_x_22Na, 10])  
print(popt_22Na)
print(pcov_22Na)

plt.figure(figsize=(12, 8))
plt.plot(x_values_22Na, y_values_22Na, 'b+:', label='data')
plt.plot(x_values_22Na, gaussiana(x_values_22Na, *popt_22Na), 'r-', label='fit')
plt.legend()
plt.title('Ajuste de gaussiana para 22Na')
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('P6_ajuste_22Na.png')
plt.show()

# Punto 1.2: Ajuste de gaussiana para 60Co

