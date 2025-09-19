import numpy as np
import math
import cv2
from scipy import signal
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

import streamlit as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import streamlit_webrtc as webrtc
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Monkey-patch streamlit_webrtc to use st.rerun() if st.experimental_rerun() is not available
try:
    from streamlit import experimental_rerun
    print("st.experimental_rerun is available")
    print(f"st.experimental_rerun: {st.experimental_rerun}")
except ImportError:
    print("st.experimental_rerun is NOT available")
    experimental_rerun = st.rerun
    print(f"st.rerun: {st.rerun}")

print(f"Streamlit version: {st.__version__}")


# Configuración inicial de Streamlit
st.set_page_config(layout="wide")
st.title("Estimación de Indicadores Vitales desde Video (rPPG)")
st.write("Graba un video (mínimo 30 segundos) donde tu rostro sea claramente visible para estimar frecuencia cardíaca, tasa respiratoria y HRV.")

# Mostrar información sobre el proceso
with st.expander("📊 ¿Cómo funciona el análisis rPPG?", expanded=False):
    st.markdown("""
    ### Proceso de Análisis de Indicadores Vitales mediante rPPG
    
    El análisis de fotopletismografía remota (rPPG) permite estimar indicadores vitales sin contacto físico, utilizando solo un video del rostro. El proceso sigue estos pasos:
    
    1. **Detección de rostro**: Utilizamos detección facial para identificar y seguir el rostro en cada fotograma del video.
    
    2. **Extracción de señales**: Analizamos los cambios sutiles en el color de la piel (especialmente en componentes verde y rojo) causados por el flujo sanguíneo.
    
    3. **Procesamiento de señal**: Aplicamos filtros para eliminar ruido y aislar las frecuencias relevantes para cada indicador vital.
    
    4. **Algoritmo CHROM**: Utilizamos el método de DeHaan que separa la información del flujo sanguíneo de otros cambios en la piel.
    
    5. **Estimación de indicadores**:
       - **Frecuencia Cardíaca**: Calculada a partir de los picos de la señal BVP (Blood Volume Pulse).
       - **Tasa Respiratoria**: Derivada de modulaciones de baja frecuencia en la señal BVP.
       - **HRV (Variabilidad del Ritmo Cardíaco)**: Medida de las variaciones entre latidos sucesivos.
    """)

# Inicialización del detector de rostros
detector = FaceDetector(minDetectionCon=0.7)

# --- Function Definitions ---

def read_video(video_file_path):
    """
    Reads video, detects the largest face in each frame, crops it,
    and returns the face frames along with the video's FPS.
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video at path: {video_file_path}")
        return None, None

    FS = cap.get(cv2.CAP_PROP_FPS)
    if FS <= 0:
        st.error("Error: No se pudo obtener FPS del video o es inválido. Asegúrate de que el video no esté corrupto.")
        return None, None # Return None for both face_frames and FS
    
    face_frames = []
    marked_face_frames = []
    frame_count = 0
    processed_frame_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 1

    std_height, std_width = None, None
    
    col1, col2 = st.columns(2)
    frame_display1 = col1.empty()
    frame_display2 = col2.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        frame_copy = frame.copy()
        img_with_faces, bboxs = detector.findFaces(frame_copy, draw=True)

        if bboxs:
            largest_bbox = None
            max_area = 0
            for bbox_info in bboxs:
                 x, y, w, h = bbox_info['bbox']
                 area = w * h
                 if area > max_area:
                     max_area = area
                     largest_bbox = bbox_info['bbox']
                     largest_center = bbox_info['center']

            if largest_bbox:
                x, y, w, h = largest_bbox
                y1, y2 = max(0, y), min(frame.shape[0], y + h)
                x1, x2 = max(0, x), min(frame.shape[1], x + w)

                if y2 > y1 and x2 > x1:
                    face_frame = frame[y1:y2, x1:x2]
                    
                    if std_height is None or std_width is None:
                        std_height, std_width = face_frame.shape[0], face_frame.shape[1]
                    
                    face_frame_resized = cv2.resize(face_frame, (std_width, std_height), 
                                                    interpolation=cv2.INTER_AREA)
                    
                    face_frame_rgb = cv2.cvtColor(face_frame_resized, cv2.COLOR_BGR2RGB)
                    face_frames.append(face_frame_rgb)
                    
                    cx, cy = largest_center
                    cv2.circle(img_with_faces, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    cv2.circle(img_with_faces, (x1, y1), 3, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img_with_faces, (x2, y1), 3, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img_with_faces, (x1, y2), 3, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img_with_faces, (x2, y2), 3, (0, 0, 255), cv2.FILLED)
                    
                    for i in range(1, 5):
                        pt_x = x1 + (i * w // 5)
                        cv2.circle(img_with_faces, (pt_x, y1), 3, (255, 0, 0), cv2.FILLED)
                    
                    for i in range(1, 5):
                        pt_x = x1 + (i * w // 5)
                        cv2.circle(img_with_faces, (pt_x, y2), 3, (255, 0, 0), cv2.FILLED)
                    
                    for i in range(1, 5):
                        pt_y = y1 + (i * h // 5)
                        cv2.circle(img_with_faces, (x1, pt_y), 3, (255, 0, 0), cv2.FILLED)
                    
                    for i in range(1, 5):
                        pt_y = y1 + (i * h // 5)
                        cv2.circle(img_with_faces, (x2, pt_y), 3, (255, 0, 0), cv2.FILLED)
                    
                    try:
                        cvzone.cornerRect(img_with_faces, (x, y, w, h), colorC=(0, 255, 0), colorR=(0, 0, 255))
                    except:
                        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    score = bbox_info.get('score')
                    if score:
                        try:
                            if isinstance(score, list):
                                conf_val = int(score[0] * 100)
                            elif hasattr(score, '__iter__'):
                                conf_val = int(float(score[0]) * 100)
                            else:
                                conf_val = int(float(score) * 100)
                        except (TypeError, ValueError, IndexError):
                            conf_val = 0
                            
                        cv2.putText(img_with_faces, f'{conf_val}%', (x, y-10), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)
                    marked_face_frames.append(img_with_faces_rgb)
                    
                    processed_frame_count += 1
                    
                    if processed_frame_count == 1 or processed_frame_count % 30 == 0:
                        frame_display1.image(face_frame_rgb, caption="Rostro detectado (recortado)", use_container_width=True)
                        frame_display2.image(img_with_faces_rgb, caption="Frame con puntos de referencia", use_container_width=True)

        if frame_count % 10 == 0 or frame_count == total_frames:
             progress = min(1.0, frame_count / total_frames)
             progress_bar.progress(progress)
             status_text.text(f"Procesando video... Frame {frame_count}/{total_frames if total_frames > 1 else '?'}. Frames con rostro: {processed_frame_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()

    if not face_frames:
        st.error("Error: No se detectaron rostros en el video.")
        return None, None
    
    if marked_face_frames:
        st.subheader("Ejemplo de detección facial con puntos de referencia")
        st.image(marked_face_frames[-1], caption="Último frame procesado con puntos de referencia", use_container_width=True)
    
    st.success(f"Video leído. Se procesaron {processed_frame_count} frames con rostros detectados.")
    
    face_frames_array = np.array(face_frames)
    return face_frames_array, FS

def process_rgb(frames):
    """Calculates the average RGB values for each frame."""
    RGB = []
    for frame in frames:
        if frame.size == 0:
            continue
        frame_area = frame.shape[0] * frame.shape[1]
        if frame_area > 0:
            sum_vals = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(sum_vals / frame_area)
    return np.asarray(RGB)

def extract_bvp(frames, FS):
    """Implements the CHROM algorithm for rPPG signal extraction."""
    st.markdown("### 2. Extrayendo señal rPPG mediante algoritmo CHROM")
    
    with st.expander("Detalles del proceso CHROM", expanded=False):
        st.markdown("""
        El algoritmo CHROM (de Haan & Jeanne, 2013) separa la información de color del flujo sanguíneo de otros cambios
        en la piel. El proceso implica:
        
        1. Extraer valores RGB promedio de cada fotograma de la cara
        2. Normalizar estos valores para reducir efectos de iluminación
        3. Proyectar estos valores en ejes específicos que separan la información del pulso
        4. Filtrar la señal para aislar las frecuencias relacionadas con el ritmo cardíaco (típicamente 0.7-2.5 Hz)
        """)
    
    parametros_col1, parametros_col2 = st.columns(2)
    
    LPF = 0.7
    HPF = 2.5
    
    with parametros_col1:
        st.write("**Parámetros de filtrado:**")
        st.write(f"- Frecuencia de corte inferior: {LPF} Hz (~{int(LPF*60)} BPM)")
        st.write(f"- Frecuencia de corte superior: {HPF} Hz (~{int(HPF*60)} BPM)")
    
    WinSec = 1.6
    if len(frames) < WinSec * FS:
         st.warning(f"Advertencia: Video demasiado corto ({len(frames)/FS:.2f}s) para la ventana de análisis ({WinSec}s). Resultados pueden ser imprecisos.")
         if len(frames) < FS * 1.0:
              st.error("Error: Video demasiado corto para análisis.")
              return None
         WinSec = len(frames) / FS * 0.9
    
    with parametros_col2:
        st.write("**Parámetros de ventana:**")
        st.write(f"- Longitud de ventana: {WinSec:.1f} segundos")
        st.write(f"- Frames por ventana: ~{int(WinSec * FS)}")

    status = st.empty()
    status.info("Procesando valores RGB de los frames faciales...")
    
    RGB = process_rgb(frames)
    if RGB.shape[0] < 2:
        st.error("Error: No se pudieron extraer suficientes datos RGB de los frames.")
        return None

    with st.expander("Ver datos RGB extraídos", expanded=False):
        fig_rgb, ax_rgb = plt.subplots(figsize=(10, 4))
        tiempo = np.arange(len(RGB)) / FS
        ax_rgb.plot(tiempo, RGB[:, 0], 'r-', label='Rojo', alpha=0.7)
        ax_rgb.plot(tiempo, RGB[:, 1], 'g-', label='Verde', alpha=0.7)
        ax_rgb.plot(tiempo, RGB[:, 2], 'b-', label='Azul', alpha=0.7)
        ax_rgb.set_xlabel('Tiempo (s)')
        ax_rgb.set_ylabel('Valor promedio de color')
        ax_rgb.set_title('Componentes RGB extraídos de los frames faciales')
        ax_rgb.legend()
        ax_rgb.grid(True, alpha=0.3)
        st.pyplot(fig_rgb)
        
        st.markdown("""
        Estos componentes RGB muestran variaciones causadas por:
        - Cambios en la iluminación
        - Movimientos faciales
        - Cambios en el volumen sanguíneo (nuestro objetivo)
        
        El algoritmo CHROM aísla estos últimos eliminando las variaciones no deseadas.
        """)

    FN = RGB.shape[0]
    NyquistF = FS / 2.0
    
    status.info("Diseñando filtros de señal...")

    if LPF >= HPF or HPF >= NyquistF:
         st.error(f"Error en frecuencias de filtro: LPF={LPF}, HPF={HPF}, Nyquist={NyquistF}. Ajusta las frecuencias o revisa el FPS.")
         HPF = min(HPF, NyquistF * 0.98)
         LPF = min(LPF, HPF * 0.98)
         if LPF <= 0: return None
         st.warning(f"Frecuencias de filtro ajustadas a: LPF={LPF:.2f}, HPF={HPF:.2f}")

    try:
        B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')
    except ValueError as e:
        st.error(f"Error al crear el filtro Butterworth: {e}. Verifica las frecuencias LPF={LPF}, HPF={HPF}, Nyquist={NyquistF}")
        return None

    status.info("Procesando señal con ventanas superpuestas...")

    WinL = math.ceil(WinSec * FS)
    if WinL % 2:
        WinL += 1
    WinL = min(WinL, FN)
    if WinL <= 0:
        st.error("Error: Longitud de ventana de análisis inválida.")
        return None

    NWin = math.floor((FN - WinL) / (WinL / 2)) + 1
    if NWin <= 0: NWin = 1

    S = np.zeros(FN)

    for i in range(NWin):
        WinS = int(i * WinL / 2)
        WinE = min(WinS + WinL, FN)
        WinM = WinS + (WinE - WinS) // 2

        RGB_win = RGB[WinS:WinE, :]
        if RGB_win.shape[0] < 2: continue

        RGBBase = np.mean(RGB_win, axis=0)
        if np.any(RGBBase == 0): continue
        RGBNorm = RGB_win / RGBBase

        Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
        Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]

        min_signal_length = 3 * 2 * 2
        if len(Xs) <= min_signal_length:
            continue
            
        try:
            Xf = signal.filtfilt(B, A, Xs)
            Yf = signal.filtfilt(B, A, Ys)
        except ValueError as e:
            st.warning(f"Advertencia: Error de filtrado en la ventana {i}: {e}. Saltando ventana.")
            continue

        std_Xf = np.std(Xf)
        std_Yf = np.std(Yf)
        if std_Yf == 0: continue

        Alpha = std_Xf / std_Yf
        SWin = Xf - Alpha * Yf

        hann_win = signal.windows.hann(len(SWin))
        SWin_hann = SWin * hann_win

        len1 = min(len(SWin_hann)//2, WinM - WinS)
        len2 = min(len(SWin_hann) - len1, WinE - WinM)

        if len1 > 0:
           S[WinS : WinS + len1] += SWin_hann[:len1]
        if len2 > 0:
           S[WinM : WinM + len2] += SWin_hann[len1 : len1+len2]

    status.success("Procesamiento CHROM completado con éxito")
    
    BVP = S
    return BVP

def analyze_heart_rate(BVP_signal, FS):
    """Estimates Heart Rate (BPM) from the BVP signal using peak detection."""
    st.markdown("### 3. Análisis de la señal BVP para estimar frecuencia cardíaca")
    
    if BVP_signal is None or len(BVP_signal) < FS * 2:
        return None, None

    with st.expander("Visualización de la señal BVP", expanded=True):
        fig_bvp, ax_bvp = plt.subplots(figsize=(10, 4))
        tiempo = np.arange(len(BVP_signal)) / FS
        ax_bvp.plot(tiempo, BVP_signal, 'g-', label='Señal BVP')
        ax_bvp.set_xlabel('Tiempo (s)')
        ax_bvp.set_ylabel('Amplitud')
        ax_bvp.set_title('Señal Blood Volume Pulse (BVP)')
        ax_bvp.grid(True, alpha=0.3)
        
        st.pyplot(fig_bvp)
        st.markdown("""
        Esta es la señal de volumen sanguíneo (BVP) extraída mediante el algoritmo CHROM.
        Los picos en esta señal corresponden a los latidos cardíacos. Para calcular la frecuencia cardíaca:
        
        1. Detectamos los picos en la señal
        2. Calculamos el tiempo entre picos consecutivos (intervalos RR)
        3. Convertimos estos intervalos a latidos por minuto (BPM)
        """)

    min_peak_dist = FS * (60.0 / 180.0)
    peaks, properties = signal.find_peaks(BVP_signal, distance=min_peak_dist, prominence=np.std(BVP_signal)*0.1)

    if len(peaks) < 2:
        st.warning("Advertencia: No se detectaron suficientes picos para calcular HR.")
        return None, None

    with st.expander("Detección de picos en la señal BVP", expanded=True):
        fig_peaks, ax_peaks = plt.subplots(figsize=(10, 4))
        tiempo = np.arange(len(BVP_signal)) / FS
        ax_peaks.plot(tiempo, BVP_signal, 'g-', label='Señal BVP')
        ax_peaks.plot(tiempo[peaks], BVP_signal[peaks], 'ro', label='Picos detectados')
        ax_peaks.set_xlabel('Tiempo (s)')
        ax_peaks.set_ylabel('Amplitud')
        ax_peaks.set_title('Detección de picos para cálculo de HR')
        ax_peaks.legend()
        ax_peaks.grid(True, alpha=0.3)
        st.pyplot(fig_peaks)
        
        st.write(f"Número de picos detectados: {len(peaks)}")
        
        if len(peaks) >= 2:
            intervals_sec = np.diff(peaks) / FS
            df_intervals = pd.DataFrame({
                'Intervalo': range(1, len(intervals_sec) + 1),
                'Duración (s)': intervals_sec,
                'BPM aproximado': 60 / intervals_sec
            })
            st.write("Muestra de intervalos entre picos (RR):")
            st.dataframe(df_intervals.head(10), use_container_width=True)

    peak_intervals_sec = np.diff(peaks) / FS

    valid_intervals = peak_intervals_sec[(peak_intervals_sec >= 60.0/180.0) & (peak_intervals_sec <= 60.0/40.0)]

    if len(valid_intervals) < 1:
         st.warning("Advertencia: No hay intervalos de pico válidos después del filtrado.")
         return None, None

    avg_ibi = np.mean(valid_intervals)
    if avg_ibi <= 0: return None

    heart_rate_bpm = 60.0 / avg_ibi
    
    st.info(f"""
    **Cálculo de frecuencia cardíaca:**
    - Intervalo RR promedio: {avg_ibi:.3f} segundos
    - Frecuencia cardíaca estimada: {heart_rate_bpm:.1f} BPM
    """)
    
    return heart_rate_bpm, peaks

def analyze_respiratory_rate(BVP_signal, FS):
    """Estimates Respiratory Rate (breaths/min) from the BVP signal."""
    st.markdown("### 4. Extracción de tasa respiratoria")
    
    if BVP_signal is None or len(BVP_signal) < FS * 5:
        return None

    resp_LPF = 0.1
    resp_HPF = 0.5
    NyquistF = FS / 2.0
    
    with st.expander("Proceso de estimación de respiración", expanded=False):
        st.markdown("""
        La respiración modula la señal de BVP a través de dos mecanismos principales:
        
        1. **Modulación respiratoria**: La respiración altera la presión intratorácica, afectando el retorno venoso y el gasto cardíaco.
        
        2. **Arritmia sinusal respiratoria (RSA)**: La respiración afecta la variabilidad del ritmo cardíaco a través del sistema nervioso autónomo.
        
        Para detectar estas modulaciones:
        1. Filtramos la señal BVP en el rango de frecuencias respiratorias (0.1-0.5 Hz)
        2. Detectamos picos en esta señal filtrada
        3. Calculamos la frecuencia de estos picos como tasa respiratoria
        """)
        
    st.write(f"Banda de frecuencia respiratoria: {resp_LPF}-{resp_HPF} Hz ({int(resp_LPF*60)}-{int(resp_HPF*60)} respiraciones/min)")

    if resp_LPF >= resp_HPF or resp_HPF >= NyquistF:
        st.warning(f"Advertencia: Frecuencias de filtro respiratorio inválidas. LPF={resp_LPF}, HPF={resp_HPF}, Nyquist={NyquistF}")
        return None

    try:
        B_resp, A_resp = signal.butter(2, [resp_LPF / NyquistF, resp_HPF / NyquistF], btype='bandpass')
        resp_signal = signal.filtfilt(B_resp, A_resp, BVP_signal)
    except ValueError as e:
        st.warning(f"Advertencia: Error al filtrar señal respiratoria: {e}")
        return None

    with st.expander("Visualización de la señal respiratoria", expanded=True):
        fig_resp, ax_resp = plt.subplots(figsize=(10, 4))
        tiempo = np.arange(len(resp_signal)) / FS
        ax_resp.plot(tiempo, resp_signal, 'b-', label='Señal respiratoria')
        ax_resp.set_xlabel('Tiempo (s)')
        ax_resp.set_ylabel('Amplitud')
        ax_resp.set_title('Señal respiratoria extraída de BVP')
        ax_resp.grid(True, alpha=0.3)
        st.pyplot(fig_resp)

    min_resp_peak_dist = FS * (60.0 / 30.0)
    resp_peaks, _ = signal.find_peaks(resp_signal, distance=min_resp_peak_dist, prominence=np.std(resp_signal)*0.2)

    if len(resp_peaks) < 2:
        st.warning("Advertencia: No se detectaron suficientes picos para calcular la tasa respiratoria.")
        return None
        
    with st.expander("Detección de picos respiratorios", expanded=True):
        fig_resp_peaks, ax_resp_peaks = plt.subplots(figsize=(10, 4))
        tiempo = np.arange(len(resp_signal)) / FS
        ax_resp_peaks.plot(tiempo, resp_signal, 'b-', label='Señal respiratoria')
        ax_resp_peaks.plot(tiempo[resp_peaks], resp_signal[resp_peaks], 'ro', label='Respiraciones detectadas')
        ax_resp_peaks.set_xlabel('Tiempo (s)')
        ax_resp_peaks.set_ylabel('Amplitud')
        ax_resp_peaks.set_title('Detección de picos respiratorios')
        ax_resp_peaks.legend()
        ax_resp_peaks.grid(True, alpha=0.3)
        st.pyplot(fig_resp_peaks)
        
        st.write(f"Número de respiraciones detectadas: {len(resp_peaks)}")

    resp_intervals_sec = np.diff(resp_peaks) / FS

    valid_resp_intervals = resp_intervals_sec[(resp_intervals_sec >= 60.0/30.0) & (resp_intervals_sec <= 60.0/6.0)]

    if len(valid_resp_intervals) < 1:
        st.warning("Advertencia: No hay intervalos respiratorios válidos después del filtrado.")
        return None

    avg_resp_interval = np.mean(valid_resp_intervals)
    if avg_resp_interval <= 0: return None

    respiratory_rate = 60.0 / avg_resp_interval
    
    st.info(f"""
    **Cálculo de tasa respiratoria:**
    - Intervalo respiratorio promedio: {avg_resp_interval:.3f} segundos
    - Tasa respiratoria estimada: {respiratory_rate:.1f} respiraciones/min
    """)
    
    return respiratory_rate

def calculate_hrv(peaks, FS):
    """Calculates HRV metrics (SDNN and RMSSD) from peak locations."""
    st.markdown("### 5. Análisis de la variabilidad del ritmo cardíaco (HRV)")
    
    with st.expander("¿Qué es el HRV y cómo se interpreta?", expanded=False):
        st.markdown("""
        La variabilidad de la frecuencia cardíaca (HRV) mide las variaciones en el tiempo entre latidos cardíacos consecutivos.
        Es un indicador importante de la salud y el funcionamiento del sistema nervioso autónomo.
        
        **Métricas principales de HRV:**
        
        - **SDNN (Desviación estándar de intervalos NN)**: Refleja todos los componentes cíclicos de la variabilidad.
        Valores más altos suelen indicar mejor salud cardiovascular y mayor capacidad de adaptación del sistema nervioso autónomo.
        
        - **RMSSD (Raíz cuadrada del valor cuadrático medio de las diferencias sucesivas)**: Refleja la actividad parasimpática.
        Valores más altos indican mayor tono vagal y mejor capacidad de recuperación.
        
        **Interpretación general:**
        - HRV alto → Mayor adaptabilidad fisiológica, mejor recuperación, menor estrés
        - HRV bajo → Posible indicador de estrés, fatiga, o problemas de salud
        """)
    
    if peaks is None or len(peaks) < 3:
        st.warning("Advertencia: No hay suficientes picos para calcular HRV de manera fiable.")
        return None, None

    rr_intervals_ms = (np.diff(peaks) / FS) * 1000

    rr_intervals_ms_filtered = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]

    if len(rr_intervals_ms_filtered) < 2:
         st.warning("Advertencia: No hay suficientes intervalos RR válidos para calcular HRV.")
         return None, None
    
    with st.expander("Visualización de intervalos RR para HRV", expanded=True):
        fig_hrv, (ax_rr, ax_diff) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax_rr.plot(np.arange(len(rr_intervals_ms_filtered)), rr_intervals_ms_filtered, 'b-o', alpha=0.6)
        ax_rr.set_ylabel('Intervalo RR (ms)')
        ax_rr.set_title('Intervalos RR para análisis de HRV')
        ax_rr.grid(True, alpha=0.3)
        
        if len(rr_intervals_ms_filtered) > 1:
            rr_diff = np.diff(rr_intervals_ms_filtered)
            ax_diff.plot(np.arange(len(rr_diff)), rr_diff, 'r-o', alpha=0.6)
            ax_diff.set_xlabel('Número de latido')
            ax_diff.set_ylabel('Diferencia RR (ms)')
            ax_diff.set_title('Diferencias sucesivas entre intervalos RR')
            ax_diff.grid(True, alpha=0.3)
            
        st.pyplot(fig_hrv)
        
        stats_df = pd.DataFrame({
            'Estadística': ['Media', 'Mediana', 'Mínimo', 'Máximo', 'Desviación estándar'],
            'Valor (ms)': [
                np.mean(rr_intervals_ms_filtered).round(2),
                np.median(rr_intervals_ms_filtered).round(2),
                np.min(rr_intervals_ms_filtered).round(2),
                np.max(rr_intervals_ms_filtered).round(2),
                np.std(rr_intervals_ms_filtered).round(2)
            ]
        })
        st.write("Estadísticas de los intervalos RR:")
        st.dataframe(stats_df, use_container_width=True)

    sdnn = np.std(rr_intervals_ms_filtered)

    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals_ms_filtered))))
    
    st.info(f"""
    **Métricas de HRV calculadas:**
    - SDNN: {sdnn:.2f} ms (Variabilidad total)
    - RMSSD: {rmssd:.2f} ms (Actividad parasimpática)
    
    *Nota: Para mediciones más precisas de HRV se recomienda un registro más largo (al menos 5 minutos).*
    """)

    return sdnn, rmssd

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.recording = False
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.frames.append(img)
        return frame

def display_results(heart_rate, respiratory_rate, sdnn, rmssd, tmp_file_path):
    """Displays the results in a dashboard format."""
    st.markdown("---")
    st.markdown("## Resultados del Análisis")
    
    st.markdown(f"*Análisis completado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*")
    
    st.subheader("Dashboard de Indicadores Vitales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if heart_rate is not None:
            if heart_rate < 60:
                interpretation = "Bradicardia (baja)"
                color = "blue"
            elif heart_rate > 100:
                interpretation = "Taquicardia (alta)"
                color = "red"
            else:
                interpretation = "Normal"
                color = "green"
            
            st.markdown(f"### ❤️ Frecuencia Cardíaca")
            st.markdown(f"<h2 style='color:{color};'>{heart_rate:.1f} BPM</h2>", unsafe_allow_html=True)
            st.markdown(f"*Interpretación: {interpretation}*")
        else:
            st.error("No se pudo estimar la Frecuencia Cardíaca.")

    with col2:
        if respiratory_rate is not None:
            if respiratory_rate < 12:
                interpretation = "Bradipnea (baja)"
                color = "blue"
            elif respiratory_rate > 20:
                interpretation = "Taquipnea (alta)"
                color = "red"
            else:
                interpretation = "Normal"
                color = "green"
            
            st.markdown(f"### 🌬️ Tasa Respiratoria")
            st.markdown(f"<h2 style='color:{color};'>{respiratory_rate:.1f} resp/min</h2>", unsafe_allow_html=True)
            st.markdown(f"*Interpretación: {interpretation}*")
        else:
            st.error("No se pudo estimar la Tasa Respiratoria.")

    with col3:
        if sdnn is not None and rmssd is not None:
            if sdnn < 20:
                interpretation = "Reducida"
                color = "orange"
            elif sdnn > 100:
                interpretation = "Elevada"
                color = "green"
            else:
                interpretation = "Normal"
                color = "blue"
            
            st.markdown(f"### ⏱️ Variabilidad Cardíaca")
            st.markdown(f"<h2 style='color:{color};'>SDNN: {sdnn:.1f} ms</h2>", unsafe_allow_html=True)
            st.markdown(f"RMSSD: {rmssd:.1f} ms")
            st.markdown(f"*Interpretación: {interpretation}*")
        else:
            st.error("No se pudo estimar el HRV.")
                        
    with st.expander("Notas sobre interpretación de resultados", expanded=False):
        st.markdown("""
        ### Interpretación de Resultados

        **Frecuencia Cardíaca (FC)**:
        - **Normal**: 60-100 BPM en adultos en reposo
        - **Bradicardia**: < 60 BPM
        - **Taquicardia**: > 100 BPM
        
        **Tasa Respiratoria**:
        - **Normal**: 12-20 respiraciones/min en adultos en reposo
        - **Bradipnea**: < 12 respiraciones/min
        - **Taquipnea**: > 20 respiraciones/min
        
        **Variabilidad del Ritmo Cardíaco (HRV)**:
        - **SDNN**: Valores típicos 20-100 ms en grabaciones cortas
        - **RMSSD**: Valores típicos 15-40 ms
        
        **Importante**: Estos resultados son estimaciones basadas en procesamiento de video y no deben utilizarse para diagnóstico médico. La precisión puede verse afectada por la calidad del video, iluminación y movimientos. Para una evaluación precisa, consulte a un profesional de la salud.
        """)
    
    if heart_rate is not None or respiratory_rate is not None or sdnn is not None:
        st.markdown("### Exportar Resultados")
        
        results_data = {
            "Indicador": ["Frecuencia Cardíaca", "Tasa Respiratoria", "HRV (SDNN)", "HRV (RMSSD)"],
            "Valor": [
                f"{heart_rate:.1f} BPM" if heart_rate is not None else "No disponible",
                f"{respiratory_rate:.1f} resp/min" if respiratory_rate is not None else "No disponible",
                f"{sdnn:.1f} ms" if sdnn is not None else "No disponible",
                f"{rmssd:.1f} ms" if rmssd is not None else "No disponible"
            ],
            "Timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 4,
            "Video": [os.path.basename(tmp_file_path)] * 4
        }
        
        results_df = pd.DataFrame(results_data)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar resultados como CSV",
            data=csv,
            file_name=f"resultados_rppg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# --- Streamlit UI ---
st.markdown("""
**IMPORTANTE:** El análisis requiere un video de al menos 30 segundos. Haz clic en "Empezar análisis" para grabar un video con tu cámara web.
""")

start = st.button("Empezar análisis")

if 'show_camera' not in st.session_state:
    st.session_state['show_camera'] = False
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if 'video_processed' not in st.session_state:
    st.session_state['video_processed'] = False

if start:
    st.session_state['show_camera'] = True
    st.session_state['recording'] = False
    st.session_state['video_processed'] = False

if st.session_state['show_camera']:
    st.info("Haz clic en 'Iniciar grabación' para comenzar (mínimo 30 segundos). Cuando termines, haz clic en 'Detener grabación' y espera el análisis.")
    ctx = webrtc_streamer(
        key="video-recorder",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=VideoRecorder,
        async_processing=True,
    )
    if ctx.video_processor:
        if not st.session_state['recording'] and not st.session_state['video_processed']:
            if st.button("Iniciar grabación"):
                ctx.video_processor.frames = []
                ctx.video_processor.recording = True
                st.session_state['recording'] = True
                st.warning("Grabando... Mantén tu rostro visible y estable.")
        if st.session_state['recording']:
            if st.button("Detener grabación"):
                ctx.video_processor.recording = False
                st.session_state['recording'] = False
                st.session_state['video_processed'] = True
                st.success("¡Video grabado! Procesando...")
                frames = ctx.video_processor.frames
                if len(frames) > 0:
                    height, width, _ = frames[0].shape
                    tmp_file_path = tempfile.mktemp(suffix='.mp4')
                    out = cv2.VideoWriter(tmp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
                    for f in frames:
                        out.write(f)
                    out.release()
                    cap = cv2.VideoCapture(tmp_file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    if duration < 29.5:
                        st.warning(f"El video grabado dura solo {duration:.1f} segundos. Por favor, graba un video de al menos 30 segundos.")
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                        st.session_state['show_camera'] = True
                        st.session_state['video_processed'] = False
                    else:
                        st.info(f"Video capturado ({duration:.1f} segundos). Procesando...")
                        proceso_bar = st.progress(0, text="Leyendo video y detectando rostro...")
                        st.markdown("## Proceso de Análisis rPPG")
                        with st.spinner('Leyendo video y detectando rostro...'):
                            face_frames, FS = read_video(tmp_file_path)
                        proceso_bar.progress(0.33, text="Extrayendo señal rPPG (CHROM)...")
                        if face_frames is not None and FS is not None:
                            with st.spinner('Extrayendo señal rPPG (CHROM)...'):
                                BVP_signal = extract_bvp(face_frames, FS)
                            proceso_bar.progress(0.66, text="Calculando indicadores vitales...")
                            if BVP_signal is not None:
                                with st.spinner('Calculando indicadores vitales...'):
                                    heart_rate, peaks = analyze_heart_rate(BVP_signal, FS)
                                    respiratory_rate = analyze_respiratory_rate(BVP_signal, FS)
                                    sdnn, rmssd = calculate_hrv(peaks, FS)
                                proceso_bar.progress(1.0, text="¡Análisis completado!")
                                
                                display_results(heart_rate, respiratory_rate, sdnn, rmssd, tmp_file_path)
                        else:
                            proceso_bar.progress(1.0, text="Fallo en la lectura del video o detección de rostro.")
                            st.error("Fallo en la lectura del video o detección de rostro. No se puede continuar.")
                        if os.path.exists(tmp_file_path):
                            try:
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                st.warning(f"No se pudo eliminar el archivo temporal: {tmp_file_path}. Error: {e}")
                        st.info("Proceso completado.")
                        st.session_state['show_camera'] = False
                        st.session_state['video_processed'] = False
# Alternative: File upload option
st.markdown("---")
st.markdown("### Alternativa: Subir video desde archivo")
uploaded_file = st.file_uploader("O sube un video desde tu dispositivo:", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    tmp_file_path = tempfile.mktemp(suffix='.mp4')
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture(tmp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    if duration < 29.5:
        st.warning(f"El video subido dura solo {duration:.1f} segundos. Por favor, sube un video de al menos 30 segundos.")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    else:
        st.info(f"Video cargado ({duration:.1f} segundos). Procesando...")
        proceso_bar = st.progress(0, text="Leyendo video y detectando rostro...")
        st.markdown("## Proceso de Análisis rPPG")
        with st.spinner('Leyendo video y detectando rostro...'):
            face_frames, FS = read_video(tmp_file_path)
        proceso_bar.progress(0.33, text="Extrayendo señal rPPG (CHROM)...")
        if face_frames is not None and FS is not None:
            with st.spinner('Extrayendo señal rPPG (CHROM)...'):
                BVP_signal = extract_bvp(face_frames, FS)
            proceso_bar.progress(0.66, text="Calculando indicadores vitales...")
            if BVP_signal is not None:
                with st.spinner('Calculando indicadores vitales...'):
                    heart_rate, peaks = analyze_heart_rate(BVP_signal, FS)
                    respiratory_rate = analyze_respiratory_rate(BVP_signal, FS)
                    sdnn, rmssd = calculate_hrv(peaks, FS)
                proceso_bar.progress(1.0, text="¡Análisis completado!")
                
                display_results(heart_rate, respiratory_rate, sdnn, rmssd, tmp_file_path)
        else:
            proceso_bar.progress(1.0, text="Fallo en la lectura del video o detección de rostro.")
            st.error("Fallo en la lectura del video o detección de rostro. No se puede continuar.")
        
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"No se pudo eliminar el archivo temporal: {tmp_file_path}. Error: {e}")

else:
    st.markdown("""
    ## Instrucciones

    1. Haz clic en el botón "Empezar análisis" para grabar un video con tu cámara web.
    2. O sube un video desde tu dispositivo usando la opción de arriba.
    3. Asegúrate de que tu rostro esté bien iluminado y visible durante al menos 30 segundos.
    4. Mantén la cabeza estable y mira hacia la cámara.
    5. El análisis comenzará automáticamente después de detener la grabación o subir el archivo.
    """)