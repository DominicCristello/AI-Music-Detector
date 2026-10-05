# Need Librosa Library to analyze audio features and load music
# Might need pandas or numpy libraries for analyzing massive amounts of data storage in a song
# Finally, Scikit-learn to decide whether track is AI or human made
# Tkinter to make visuals


#Needs Machine Learning

#should be a turnitin but for music, where you can upload a track and it will analyze the audio features and
# compare them to a database of known AI generated music to determine if the track is likely AI generated or human made.


import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pylab as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import pygame
from itertools import cycle
from glob import glob
import seaborn as sns


# Custom Modules
from computeinfo import compute_info
#from process_folder import process_folder



sns.set_theme(style="white", palette = None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files = glob("*.mp3") # or whatever file type you want to analyze
pygame.mixer.init()




class AudioApp:

    #also, in the future calculate spectral rolloff
    def __init__(self, root):
        self.root = root
        self.root.title("Waveform + Spectrogram + Info")
        self.root.geometry("950x800")

        # Make rows/columns expandable
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=6)  # plots
        root.rowconfigure(1, weight=0)  # buttons
        root.rowconfigure(2, weight=3)  # info
        root.rowconfigure(3, weight=0)  # progress
        root.rowconfigure(4, weight=0)  # status

        # ---- Matplotlib Figure ----
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(2, 1, figsize=(9, 6))

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, sticky="nsew", padx=10, pady=10
        )

        # ---- Buttons ----
        button_frame = tk.Frame(root)
        button_frame.grid(row=1, column=0, pady=5)

        tk.Button(button_frame, text="Load Audio", command=self.load_audio).grid(
            row=0, column=0, padx=5
        )
        tk.Button(button_frame, text="Play", command=self.play_audio).grid(
            row=0, column=1, padx=5
        )
        tk.Button(button_frame, text="Stop", command=self.stop_audio).grid(
            row=0, column=2, padx=5
        )

        # ---- Info Box ----
        self.info_box = tk.Text(root, wrap="word", height=10)
        self.info_box.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.info_box.insert("1.0", "Load a file to see its information here.")
        self.info_box.configure(state="disabled")

        # ---- Progress Bar ----
        self.progress = ttk.Progressbar(
            root, orient="horizontal", length=400, mode="determinate", maximum=100
        )
        self.progress.grid(row=3, column=0, pady=5)

        # ---- Status Label ----
        self.status = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status, anchor="w").grid(
            row=4, column=0, sticky="ew", padx=10, pady=(0, 10)
        )

        self.file_path = None
    def set_info_text(self, text):
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", text)
        self.info_box.configure(state="disabled")

    def load_audio(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*")])
        if not self.file_path:
            return
    # We will load the audio file, extract features, and update the GUI in stages to provide feedback to the user.


        try:

            self.update_progress(10, "Loading audio file...")
            y, sr = librosa.load(self.file_path, sr=None, mono=True)
            self.update_progress(30, "Drawing waveform...")
            self.ax_wave.clear()
            librosa.display.waveshow(y, sr=sr, ax=self.ax_wave)
            self.ax_wave.set_title("Waveform")

            self.update_progress(55, "Generating spectrogram...")
            self.ax_spec.clear()
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=self.ax_spec)
            self.ax_spec.set_title("Mel Spectrogram (dB)")

            self.update_progress(75, "Updating display...")
            self.fig.tight_layout()
            self.canvas.draw()

            self.update_progress(90, "Computing track info (this could take a while...)")
            info_text = compute_info(y, sr, filename=os.path.basename(self.file_path))
            self.set_info_text(info_text)

            self.update_progress(100, "Preparing playback...")
            pygame.mixer.music.load(self.file_path)

            self.status.set("Done.")
            self.root.after(500, lambda: self.progress.configure(value=0))
            filename = os.path.basename(self.file_path)
            self.set_info_text(f"Loaded: {filename}\n\n{info_text}")

        except Exception as e:
            self.status.set(f"Load failed: {e}")
            self.set_info_text(f"ERROR while loading:\n{e}")



    def play_audio(self):

        if not self.file_path:
            self.status.set("No file loaded.")
            return
        try:
            pygame.mixer.music.play()
            self.status.set("Playing...")
        except Exception as e:
            self.status.set(f"Play failed: {e}")
            self.set_info_text(f"ERROR while playing:\n{e}")

    def stop_audio(self):
        try:
            pygame.mixer.music.stop()
            self.status.set("Stopped.")
        except Exception as e:
            self.status.set(f"Stop failed: {e}")
            self.set_info_text(f"ERROR while stopping:\n{e}")


    def update_progress(self, value, message):
        self.progress["value"] = value
        self.status.set(message)
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
    # # --- RUN THE BATCH PROCESS DIRECTLY ---
    # print("--- STARTING AI DATASET AUDIT ---")
    # # Make sure you change 'label' to 1 inside process_folder for this run
    # process_folder("./F:/AI_Music_Detection_Library/AI_Music", "AI_Dataset.csv")
    # #THEN CHANGE LABEL TO 0 AND RUN AGAIN FOR HUMAN DATASET
    # print("--- BATCH COMPLETE ---")


#ipd.Audio(audio_files[0]) # to play the audio file in jupyter notebook
#y, sr = librosa.load(audio_files[0])

# y is raw data of audio file, sr is integer value of sample rate of audio file
#to graph:
#pd.Series(y).plot(figsize=(10,5), lw=1, title = "Audio Sample", color = color_pal[0])
#plt.show()
#to get rid of extra space at beginning and end of audio file (_ is index which we don't need):
#y_trimmed, _ = librosa.effects.trim(y)

#for a spectrogram, apply a fourier transform (stft(y)):
#D = librosa.stft(y)
#S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#now plot the data:
#fig, ax = plt(subplots(figsize=(10,5 )))
#img = librosa.display.specshow(S_db, x_axis='time', y_axis='log',ax=ax)
#ax.set_title('Spectogram Example', fontsize = 20)
#fig.color(img, ax=ax, format=f'%0.2f')

#melodic spectogram

#S = librosa.feature.melspectogram(y, sr=sr, n_mels=128)
#now apply same transform:
#S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
#plot the mel spectogram:


#fig, ax = plt.subplots(figsize=(10,5 ))
#img = librosa.display.specshow(S_db_mel, x_axis='time', y_axis='log',ax=ax)
#ax.set_title('Mel Spectogram Example', fontsize = 20)
#fig.color(img, ax=ax, format=f'%0.2f')
