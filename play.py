import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import time
from datetime import datetime
import os
import ale_py 
from typing import Callable, Any
import torch
from src.utils import preprocess_observation
from src.trainning.dqn_train import DQNNetwork
# dqn_model = DQNNetwork(action_space=6).cuda()
# dqn_model.load_state_dict(torch.load("dqn_galaxian.pt"))
# dqn_model.eval()

# def dqn_policy(obs, action_space):
#     state = preprocess_observation(obs)
#     with torch.no_grad():
#         q_values = dqn_model(torch.tensor(state).unsqueeze(0).cuda())
#         return torch.argmax(q_values).item()


STUDENT_EMAIL_PREFIX = "alo20172"

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
    # De forma que galaxian se mueve de izquierda a derecha y dispara dado que el objetivo es eliminar naves enemigas.
    # Noop es que se mantiene en el mismo estado sin hacer nada.
    relevant_actions = [0, 1, 2, 3, 4, 5]
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
    VIDEO_FOLDER = "videos_demo"
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    temp_video_path = os.path.join(VIDEO_FOLDER, "temp_video_recording.mp4")

    env = gym.make(
        "ALE/Galaxian-v5", 
        render_mode="rgb_array", 
        full_action_space=True 
    )
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

    # Ejecución del Episodio
    print(f"Iniciando grabación del episodio en {env.spec.id}...")
    
    timestamp_inicio = datetime.now()

    while not done and not truncated:
        action = policy(observation, env.action_space) 
        
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episodio finalizado. Puntuación alcanzada: {total_reward}")

    # Cerrar el Entorno y Finalizar la Grabación
    env.close()
    
    # Renombrar el Video con el Formato Requerido
    timestamp_str = timestamp_inicio.strftime("%Y%m%d%H%M")
    
    final_filename = (
        f"{STUDENT_EMAIL_PREFIX}_{timestamp_str}_{int(total_reward)}.mp4"
    )
    final_filepath = os.path.join(VIDEO_FOLDER, final_filename)
    
    # Buscar el archivo generado por el wrapper (tiene un timestamp en su nombre)
    generated_files = [f for f in os.listdir(VIDEO_FOLDER) if f.startswith("temp_video_recording")]
    
    if generated_files:
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