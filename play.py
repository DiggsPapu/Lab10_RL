# play.py (CÓDIGO CORREGIDO Y COMPLETO)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import time
from datetime import datetime
import os
import ale_py # Asegura el registro del namespace ALE/
from typing import Callable, Any

# --- Configuración del Estudiante ---
STUDENT_EMAIL_PREFIX = "alo20172"
# --- Fin Configuración del Estudiante ---

def default_policy(observation: Any, action_space: gym.spaces.Discrete) -> int:
    """
    Política de ejemplo: Elige una acción aleatoria.
    En un proyecto real, esto sería reemplazado por el agente entrenado (ej. DQN).
    
    Args:
        observation (Any): El estado actual del entorno.
        action_space (gym.spaces.Discrete): El espacio de acciones del entorno.
        
    Returns:
        int: La acción a tomar.
    """
    # Acciones relevantes para Galaxian:
    # 0: NOOP, 1: FIRE, 3: RIGHT, 4: LEFT, 11: RIGHTFIRE, 12: LEFTFIRE
    relevant_actions = [0, 1, 3, 4, 11, 12]
    return np.random.choice(relevant_actions)

def record_episode(policy: Callable[[Any, gym.spaces.Discrete], int]) -> str:
    """
    Ejecuta un episodio completo en ALE/Galaxian-v5, registrándolo
    visualmente y guardando el resultado en formato MP4 con la
    nomenclatura requerida.

    Args:
        policy (Callable): Una función que toma (observación, action_space)
                           y retorna la acción (int) a ejecutar.

    Returns:
        str: El nombre del archivo de video generado.
    """
    # 1. Configuración del Video
    VIDEO_FOLDER = "videos_demo"
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    # El nombre de archivo inicial es temporal. Lo renombraremos al final.
    temp_video_path = os.path.join(VIDEO_FOLDER, "temp_video_recording.mp4")

    # 2. Inicialización del Entorno con Wrapper de Grabación
    # *** CORRECCIÓN CLAVE: full_action_space=True ***
    # Esto asegura que el entorno acepte acciones con índices 11 y 12.
    env = gym.make(
        "ALE/Galaxian-v5", 
        render_mode="rgb_array", 
        full_action_space=True 
    )
    # ---------------------------------------------
    
    # El wrapper RecordVideo debe envolver el entorno base
    env = RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        name_prefix="temp_video_recording",
        episode_trigger=lambda x: True, # Graba el único episodio
        disable_logger=True
    )
    
    # Inicialización del episodio
    observation, info = env.reset(seed=int(time.time()))
    done = False
    truncated = False
    total_reward = 0

    # 3. Ejecución del Episodio
    print(f"Iniciando grabación del episodio en {env.spec.id}...")
    
    timestamp_inicio = datetime.now()

    while not done and not truncated:
        action = policy(observation, env.action_space) 
        
        # El wrapper RecordVideo llama a step()
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episodio finalizado. Puntuación alcanzada: {total_reward}")

    # 4. Cerrar el Entorno y Finalizar la Grabación
    # Al cerrar el entorno, RecordVideo guarda el video.
    env.close()
    
    # 5. Renombrar el Video con el Formato Requerido
    
    # Formato de timestamp: AAAAMMDDHHMM
    timestamp_str = timestamp_inicio.strftime("%Y%m%d%H%M")
    
    # Formato: <correo_estudiante>_<timestamp_episodio>_<puntuación_alcanzada>.mp4
    final_filename = (
        f"{STUDENT_EMAIL_PREFIX}_{timestamp_str}_{int(total_reward)}.mp4"
    )
    final_filepath = os.path.join(VIDEO_FOLDER, final_filename)
    
    # Buscar el archivo generado por el wrapper (tiene un timestamp en su nombre)
    generated_files = [f for f in os.listdir(VIDEO_FOLDER) if f.startswith("temp_video_recording")]
    
    if generated_files:
        # Tomar el archivo generado y renombrarlo
        actual_video_path = os.path.join(VIDEO_FOLDER, generated_files[0])
        os.rename(actual_video_path, final_filepath)
        print(f"Video guardado exitosamente como: {final_filename}")
        return final_filename
    else:
        print("Error: No se encontró el archivo de video generado.")
        return ""

if __name__ == '__main__':
    final_video_name = record_episode(policy=default_policy)
    
    if final_video_name:
        print(f"\n--- Tarea Completada ---")
        print(f"El archivo '{final_video_name}' es su entregable de video.")