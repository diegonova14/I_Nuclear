import numpy as np

# Load data from .dat files
# 600 seconds

t600_22Na = np.genfromtxt('Practica 1 y 2/NaI_22Na_600s.dat', names=True)
t600_60Co = np.genfromtxt('Practica 1 y 2/NaI_60Co_600s.dat', names=True)
t600_137Cs = np.genfromtxt('Practica 1 y 2/NaI_137Cs_600s.dat', names=True)
t600_fondo = np.genfromtxt('Practica 1 y 2/NaI_Fondo_600s.dat', names=True)

# 300 seconds
t300_57Co = np.genfromtxt('Practica 1 y 2/NaI_57Co_300s.dat', names=True)
t300_fondo = np.genfromtxt('Practica 1 y 2/NaI_Fondo_300s.dat', names=True)

#punto 1: graficar espectros
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.plot(t600_22Na[:,0], t600_22Na[:,1], label='22Na')
plt.plot(t600_60Co[:,0], t600_60Co[:,1], label='60Co')
plt.plot(t600_137Cs[:,0], t600_137Cs[:,1], label='137Cs')
plt.plot(t600_fondo[:,0], t600_fondo[:,1], label='Fondo')
plt.xlabel('Canal')
plt.ylabel('Cuentas')
plt.legend()
plt.savefig('espectros600.png')
plt.show()


