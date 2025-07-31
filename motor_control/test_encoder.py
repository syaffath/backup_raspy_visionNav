from gpiozero import Button
import time

count = 0

def on_pulse():
    global count
    count += 1
    print(f"Pulse: {count}")

encoder = Button(17)   # ganti 22 untuk kanan
encoder.when_pressed = on_pulse  # atau coba when_released

input("Siapkan posisi awal roda (tempel stiker). Tekan Enter jika siap...")
count = 0

print("Putar roda 1 putaran penuh pelan-pelan...")
input("Tekan Enter setelah selesai 1 putaran.")

print(f"Total pulse untuk 1 putaran: {count}")
