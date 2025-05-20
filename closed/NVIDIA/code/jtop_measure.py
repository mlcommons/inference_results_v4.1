import time
from jtop import jtop
import threading
with jtop() as jetson :
    class JtopMeasure:
        def __init__(self):
            self.jetson = jetson
            self.running = False
            self.thread = None

        def start(self):
            if not self.jetson.ok():
                print("Erreur lors de l'initialisation de jtop.")
                return
            print("###################Debut de la mesure jtop###########################")
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

        def stop(self):
            self.running = False
            print("###################Fin de la mesure jtop###########################")
            if self.thread is not None:
                self.thread.join()

        def run(self):
            with open("/media/nvidia/177d5801-095d-441b-88e2-959056c30fac/data/consommation_energie_jetson.csv", "w") as f:
                f.write("timestamp,gpu_power\n")
                start_time = time.time()  # Temps de départ
                elapsed_time = 0  # Temps écoulé initialisé à 0

                try:
                    while self.running and self.jetson.ok():
                        # Lire les données de puissance du GPU
                        gpu_power = self.jetson.power[1]["GPU"]["cur"]
                        f.write(f"{elapsed_time},{gpu_power}\n")
                        f.flush()  # Forcer l'écriture immédiate sur le disque
                        elapsed_time = time.time() - start_time  # Mettre à jour le temps écoulé
                except Exception as e:
                    print(f"Une erreur s'est produite : {e}")
