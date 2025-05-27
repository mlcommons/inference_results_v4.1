import time
from jtop import jtop
import threading
import traceback
print(float(jtop.interval))
class JtopMeasure:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JtopMeasure, cls).__new__(cls)
            cls._instance.running = False
            cls._instance.thread = None
        return cls._instance

    def start(self):
        if not self.running:
            print("###################Début de la mesure jtop###########################")
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
        else:
            print("Le processus est déjà en cours d'exécution.")

    def stop(self):
        if self.running:
            self.running = False
            print("###################Fin de la mesure jtop###########################")
            if self.thread is not None:
                self.thread.join()
        else:
            print("Le processus n'est pas en cours d'exécution.")

    def run(self):
        try:
            with jtop() as jetson:
                if not jetson.ok():
                    print("Erreur lors de l'initialisation de jtop.")
                    return

                with open("/media/nvidia/00640565-37a8-4b58-a27b-fbd90cd43fec/data/consommation_energie_jetson.csv", "w") as f:
                    f.write("timestamp,gpu_power\n")
                    
                    start_time = time.time()

                    while self.running:
                        gpu_power = jetson.power['rail']['VDD_GPU_SOC']['power']
                        elapsed_time = time.time() - start_time
                        f.write(f"{elapsed_time},{gpu_power}\n")
                        f.flush()
                        time.sleep(0.2)
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")
            traceback.print_exc()
