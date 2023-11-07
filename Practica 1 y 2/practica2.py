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
p_electron = 511

max_i_22Na = y_values_22Na[400:].index(max(y_values_22Na[400:])) + 400
max_x_22Na = x_values_22Na[max_i_22Na]

max_i_22Na_e = y_values_22Na[100:200].index(max(y_values_22Na[100:200])) + 100
max_x_22Na_e = x_values_22Na[max_i_22Na_e]

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
plt.axvline(x=max_x_22Na_e, color='r')
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
plt.text(max_x_22Na_e, max(y_values_22Na[100:200]), "{:.1f} keV".format(p_electron), rotation=90)
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
from numpy import trapz
import pandas as pd

def gaussiana(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

x_values_22Na = np.array(x_values_22Na)
y_values_22Na = np.array(y_values_22Na)

popt_22Na, pcov_22Na = curve_fit(gaussiana, x_values_22Na, y_values_22Na, p0=[max(y_values_22Na), max_x_22Na, 10])  
print(f"Parametros gaussiana 22Na: {popt_22Na[0]:.1f}, {popt_22Na[1]:.1f}, {popt_22Na[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_22Na)

popt_22Na_e, pcov_22Na_e = curve_fit(gaussiana, x_values_22Na, y_values_22Na, p0=[max(y_values_22Na[100:200]), max_x_22Na_e, 10])
print(f"Parametros gaussiana 22Na_e: {popt_22Na_e[0]:.1f}, {popt_22Na_e[1]:.1f}, {popt_22Na_e[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_22Na_e)

def fwhm(x, y):
    half_max = max(y) / 2.
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - y[])
    d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
    # find the left and right most indexes
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[0][0]
    return x[right_idx] - x[left_idx]

f_22Na_1 = gaussiana(x_values_22Na, *popt_22Na_e)
f_22Na_2 = gaussiana(x_values_22Na, *popt_22Na)

fwhm_22Na_1 = fwhm(x_values_22Na, f_22Na_1)
print(f"El valor de FWHM para la gaussiana f_22Na_1 es: {fwhm_22Na_1:.2f}")
fwhm_22Na_2 = fwhm(x_values_22Na, f_22Na_2)
print(f"El valor de FWHM para la gaussiana f_22Na_2 es: {fwhm_22Na_2:.2f}")

plt.figure(figsize=(12, 8))
plt.plot(x_values_22Na, y_values_22Na, 'b+:', label='Datos')
plt.plot(x_values_22Na, f_22Na_1, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para el primer pico 22Na: {p_electron:.1f}  keV')
plt.xlim([popt_22Na_e[1]-4*popt_22Na_e[2], popt_22Na_e[1]+4*popt_22Na_e[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_22Na_1.png')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(x_values_22Na, y_values_22Na, 'b+:', label='Datos')
plt.plot(x_values_22Na, f_22Na_2, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para el segundo pico 22Na: {p_22Na_1:.1f}  keV')
plt.xlim([popt_22Na[1]-4*popt_22Na[2], popt_22Na[1]+4*popt_22Na[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_22Na_2.png')
plt.show()


integral_f_22Na_1 = trapz(f_22Na_1, x_values_22Na)
print(f"La intensidad para el primer pico de 22Na es: {integral_f_22Na_1}")
integral_f_22Na_2 = trapz(f_22Na_2, x_values_22Na)
print(f"La intensidad para el segundo pico de 22Na es: {integral_f_22Na_2}")

# Punto 1.2: Ajuste de gaussiana para 60Co

x_values_60Co = np.array(x_values_60Co)
y_values_60Co = np.array(y_values_60Co)

popt_60Co_1, pcov_60Co_1 = curve_fit(gaussiana, x_values_60Co, y_values_60Co, p0=[max(y_values_60Co[300:400]), max_x_60Co_1, 10])
print(f"Parametros gaussiana 60Co_1: {popt_60Co_1[0]:.1f}, {popt_60Co_1[1]:.1f}, {popt_60Co_1[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_60Co_1)

popt_60Co_2, pcov_60Co_2 = curve_fit(gaussiana, x_values_60Co, y_values_60Co, p0=[max(y_values_60Co[400:]), max_x_60Co_2, 10])
print(f"Parametros gaussiana 60Co_2: {popt_60Co_2[0]:.1f}, {popt_60Co_2[1]:.1f}, {popt_60Co_2[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_60Co_2)

f_60Co_1 = gaussiana(x_values_60Co, *popt_60Co_1)
f_60Co_2 = gaussiana(x_values_60Co, *popt_60Co_2)
fit_60Co = f_60Co_1 + f_60Co_2

fwhm_60Co_1 = fwhm(x_values_60Co, f_60Co_1)
print(f"El valor de FWHM para la gaussiana f_60Co_1 es: {fwhm_60Co_1:.2f}")
fwhm_60Co_2 = fwhm(x_values_60Co, f_60Co_2)
print(f"El valor de FWHM para la gaussiana f_60Co_2 es: {fwhm_60Co_2:.2f}")

plt.figure(figsize=(12, 8))
plt.plot(x_values_60Co, y_values_60Co, 'b+:', label='Datos')
plt.plot(x_values_60Co, fit_60Co, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para los picos de 60Co: {p_60Co_1:.1f}  keV, {p_60Co_2:.1f}  keV')
plt.xlim([popt_60Co_1[1]-4*popt_60Co_1[2], popt_60Co_2[1]+4*popt_60Co_2[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_60Co.png')
plt.show()

integral_f_60Co_1 = trapz(f_60Co_1, x_values_60Co)
print(f"La intensidad para el primer pico de 60Co es: {integral_f_60Co_1}")
integral_f_60Co_2 = trapz(f_60Co_2, x_values_60Co)
print(f"La intensidad para el segundo pico de 60Co es: {integral_f_60Co_2}")

# Punto 1.3: Ajuste de gaussiana para 137Cs

x_values_137Cs = np.array(x_values_137Cs)
y_values_137Cs = np.array(y_values_137Cs)

popt_137Cs_1, pcov_137Cs_1 = curve_fit(gaussiana, x_values_137Cs, y_values_137Cs, p0=[max(y_values_137Cs[:100]), max_x_137Cs_1, 10])
print(f"Parametros gaussiana 137Cs_1: {popt_137Cs_1[0]:.1f}, {popt_137Cs_1[1]:.1f}, {popt_137Cs_1[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_137Cs_1)

popt_137Cs_2, pcov_137Cs_2 = curve_fit(gaussiana, x_values_137Cs, y_values_137Cs, p0=[max(y_values_137Cs[200:300]), max_x_137Cs_2, 10])
print(f"Parametros gaussiana 137Cs_2: {popt_137Cs_2[0]:.1f}, {popt_137Cs_2[1]:.1f}, {popt_137Cs_2[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_137Cs_2)

f_137Cs_1 = gaussiana(x_values_137Cs, *popt_137Cs_1)
f_137Cs_2 = gaussiana(x_values_137Cs, *popt_137Cs_2)
fit_137Cs = f_137Cs_1 + f_137Cs_2

fwhm_137Cs_1 = fwhm(x_values_137Cs, f_137Cs_1)
print(f"El valor de FWHM para la gaussiana f_137Cs_1 es: {fwhm_137Cs_1:.2f}")
fwhm_137Cs_2 = fwhm(x_values_137Cs, f_137Cs_2)
print(f"El valor de FWHM para la gaussiana f_137Cs_2 es: {fwhm_137Cs_2:.2f}")

plt.figure(figsize=(12, 8))
plt.plot(x_values_137Cs, y_values_137Cs, 'b+:', label='Datos')
plt.plot(x_values_137Cs, f_137Cs_1, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para el primer pico 137Cs: {p_137Cs_alfa:.1f}  keV')
plt.xlim([popt_137Cs_1[1]-4*popt_137Cs_1[2], popt_137Cs_1[1]+4*popt_137Cs_1[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_137Cs_1.png')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(x_values_137Cs, y_values_137Cs, 'b+:', label='Datos')
plt.plot(x_values_137Cs, f_137Cs_2, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para el segundo pico 137Cs: {p_137Cs_1:.1f}  keV')
plt.xlim([popt_137Cs_2[1]-4*popt_137Cs_2[2], popt_137Cs_2[1]+4*popt_137Cs_2[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_137Cs_2.png')
plt.show()

integral_f_137Cs_1 = trapz(f_137Cs_1, x_values_137Cs)
print(f"La intensidad para el primer pico de 137Cs es: {integral_f_137Cs_1}")
integral_f_137Cs_2 = trapz(f_137Cs_2, x_values_137Cs)
print(f"La intensidad para el segundo pico de 137Cs es: {integral_f_137Cs_2}")

# Punto 1.4: Ajuste de gaussiana para 57Co

x_values_57Co = np.array(x_values_57Co)
y_values_57Co = np.array(y_values_57Co)

popt_57Co_1, pcov_57Co_1 = curve_fit(gaussiana, x_values_57Co, y_values_57Co, p0=[max(y_values_57Co[:100]), max_x_57Co_1, 10])
print(f"Parametros gaussiana 57Co_1: {popt_57Co_1[0]:.1f}, {popt_57Co_1[1]:.1f}, {popt_57Co_1[2]:.1f} (a,x0,sigma)")
print("Matriz de covarianzas:")
print(pcov_57Co_1)

f_57Co_1 = gaussiana(x_values_57Co, *popt_57Co_1)

fwhm_57Co_1 = fwhm(x_values_57Co, f_57Co_1)
print(f"El valor de FWHM para la gaussiana f_57Co_1 es: {fwhm_57Co_1:.2f}")

plt.figure(figsize=(12, 8))
plt.plot(x_values_57Co, y_values_57Co, 'b+:', label='Datos')
plt.plot(x_values_57Co, f_57Co_1, 'r-', label='Ajuste')
plt.legend()
plt.title(f'Ajuste de gaussiana para el primer pico 57Co: {p_57Co_1:.1f}  keV')
plt.xlim([popt_57Co_1[1]-4*popt_57Co_1[2], popt_57Co_1[1]+4*popt_57Co_1[2]])
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.savefig('A1_ajuste_57Co_1.png')
plt.show()

integral_f_57Co_1 = trapz(f_57Co_1, x_values_57Co)
print(f"La intensidad para el primer pico de 57Co es: {integral_f_57Co_1}")

# Crear una tabla para colocar los valores de los parámetros popt, fwhm e integral_f

data = {'a': [popt_22Na_e[0], popt_22Na[0], popt_60Co_1[0], popt_60Co_2[0], popt_137Cs_1[0], popt_137Cs_2[0], popt_57Co_1[0]],
    'mu': [popt_22Na_e[1], popt_22Na[1], popt_60Co_1[1], popt_60Co_2[1], popt_137Cs_1[1], popt_137Cs_2[1], popt_57Co_1[1]],
    'sigma': [popt_22Na_e[2], popt_22Na[2], popt_60Co_1[2], popt_60Co_2[2], popt_137Cs_1[2], popt_137Cs_2[2], popt_57Co_1[2]],
    'FWHM': [fwhm_22Na_2, fwhm_22Na_1, fwhm_60Co_1, fwhm_60Co_2, fwhm_137Cs_1, fwhm_137Cs_2, fwhm_57Co_1],
    'I': [integral_f_22Na_2, integral_f_22Na_1, integral_f_60Co_1, integral_f_60Co_2, integral_f_137Cs_1, integral_f_137Cs_2, integral_f_57Co_1]}

df = pd.DataFrame(data, index=['22Na_1','22Na_2', '60Co_1', '60Co_2', '137Cs_1', '137Cs_2', '57Co'])
print(df)

# Punto 2: Calibración de energía