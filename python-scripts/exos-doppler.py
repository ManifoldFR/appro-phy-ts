import numpy as np
import matplotlib.pyplot as plt

def h(t):
    freq = [435,580]
    pulsa = [2*np.pi*f for f in freq]
    amp = [1.,1/3]
    phas = [0.6,0]
    return sum(am*np.cos(pu*t - ph) for (am,pu,ph) in zip(amp,pulsa,phas) 
        )
fs = 2048
N = 1000
dt = 1/fs
times = np.linspace(0,N*dt,N)
a = h(times)

fig, ax = plt.subplots(2,1, figsize=(8,8))

def make_time():
    axi = ax[0]
    axi.plot(times,a)
    axi.set_title(r"Signal $s(t)$ émis par la sirène")
    axi.set_ylabel(r"$s(t)$ (Volts)")
    axi.set_xlabel(r"Temps $t$ (s)")
    axi.grid()

def make_freq():
    axi = ax[1]
    af = np.fft.fft(a,norm="ortho")
    freqs = np.fft.fftfreq(N, d=dt)
    axi.plot(freqs,np.abs(af))
    axi.set_xlim(0,800)
    axi.set_xlabel("Fréquence $f$ (Hz)")
    axi.set_title("Spectre en fréquence")
    axi.set_ylabel(r"Amplitude $|\hat s(f)|$")
    start, end = axi.get_xlim()
    axi.xaxis.set_ticks(np.arange(start, end, 50))
    axi.grid()
    return af

make_time(); make_freq()
fig.tight_layout()
fig.show()