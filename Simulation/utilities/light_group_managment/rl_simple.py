import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from Simulation.config import Defaults
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.agents.city_structure_entities.intersection_light_group import IntersectionLightGroup

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices and Defaults.CUDA_GPU_ENABLED:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf_device = '/GPU:0'
else:
    tf_device = '/CPU:0'
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

HIDDEN_LAYERS = Defaults.SRL_HIDDEN_LAYERS
SRL_HIDDEN_LAYER_SIZE = Defaults.SRL_HIDDEN_LAYER_SIZE
UPDATE_EVERY = Defaults.SRL_UPDATE_EVERY
BATCH_SIZE   = Defaults.SRL_BATCH_SIZE
DROPOUT      = Defaults.SRL_DROPOUT

def make_policy_net(input_dim=13, hidden=SRL_HIDDEN_LAYER_SIZE):
    with tf.device(tf_device):
        inputs = tf.keras.Input(shape=(input_dim,), name="state")
        x = layers.Dense(hidden, activation='relu')(inputs)
        for _ in range(HIDDEN_LAYERS - 1):
            x = layers.Dense(hidden, activation='relu')(x)

        x = layers.LayerNormalization()(x)
        x = layers.Dropout(DROPOUT)(x)

        logits = layers.Dense(2)(x)
        return tf.keras.Model(inputs=inputs, outputs=logits)

def get_rl_state(intersection_light_group: "IntersectionLightGroup") -> list[float]:
    city = intersection_light_group.city_model
    occupancy = city.occupancy_map

    ns_coords = np.array(intersection_light_group.ns_in_coords)
    ew_coords = np.array(intersection_light_group.ew_in_coords)

    local_ns = occupancy[ns_coords[:, 1], ns_coords[:, 0]].sum() if ns_coords.size else 0
    local_ew = occupancy[ew_coords[:, 1], ew_coords[:, 0]].sum() if ew_coords.size else 0
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    nbrs = list(intersection_light_group.get_neighbor_groups().values())
    sum_n_ns = sum(getattr(n, '_pressure_ns', 0) for n in nbrs)
    sum_n_ew = sum(getattr(n, '_pressure_ew', 0) for n in nbrs)
    cnt = max(1, len(nbrs))
    avg_n_ns = sum_n_ns / cnt
    avg_n_ew = sum_n_ew / cnt

    phase_bit = [1, 0] if intersection_light_group._rl_phase == 0 else [0, 1]
    t_norm = intersection_light_group._rl_timer / getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 30)

    intersection_size = intersection_light_group.intersection_size
    penalty_score = intersection_light_group.penalty_score

    avg_penalty_intersection_size = sum(getattr(n, 'intersection_size', 0) for n in nbrs)/cnt
    avg_penalty_score = sum(getattr(n, 'penalty_score', 0) for n in nbrs)/cnt

    return [float(local_ns), float(local_ew), p_ns, p_ew, avg_n_ns, avg_n_ew] + phase_bit + [t_norm] + [intersection_size,penalty_score,avg_penalty_intersection_size, avg_penalty_score]

def run_rl_control(intersection_light_group: "IntersectionLightGroup"):
    if not hasattr(intersection_light_group, '_rl_phase'):
        intersection_light_group._rl_phase = 0
        intersection_light_group._rl_timer = 0

    occ = intersection_light_group.city_model.occupancy_map
    ns_coords = np.array(intersection_light_group.ns_in_coords)
    ew_coords = np.array(intersection_light_group.ew_in_coords)
    local_ns = occ[ns_coords[:, 1], ns_coords[:, 0]].sum() if ns_coords.size else 0
    local_ew = occ[ew_coords[:, 1], ew_coords[:, 0]].sum() if ew_coords.size else 0


    intersection_light_group._pressure_ns = int(local_ns - local_ew)
    intersection_light_group._pressure_ew = int(local_ew - local_ns)

    state = get_rl_state(intersection_light_group)
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    with tf.device(tf_device):
        logits = intersection_light_group.rl_policy(state_tensor)
        action_probs = tf.nn.softmax(logits[0])
        action = int(np.random.choice([0, 1], p=action_probs.numpy()))

    intersection_light_group._rl_timer += 1
    if intersection_light_group._rl_timer == 1:
        intersection_light_group.apply_phase(intersection_light_group._rl_phase)

    min_g = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
    if action == 1 and intersection_light_group._rl_timer >= min_g:
        intersection_light_group._rl_phase = 1 - intersection_light_group._rl_phase
        intersection_light_group._rl_timer = 0

    reward = -(local_ns + local_ew)
    next_state = get_rl_state(intersection_light_group)
    intersection_light_group.memory.append((state, int(action), reward, next_state))

    if len(intersection_light_group.memory) >= UPDATE_EVERY:
        train_rl(intersection_light_group, BATCH_SIZE)
        intersection_light_group.memory.clear()

def run_batched_rl_control(intersections: list["IntersectionLightGroup"], policy_model: tf.keras.Model):
    states = []
    batch = []

    for ig in intersections:
        occ = ig.city_model.occupancy_map
        ns_coords = np.array(ig.ns_in_coords)
        ew_coords = np.array(ig.ew_in_coords)
        local_ns = occ[ns_coords[:, 1], ns_coords[:, 0]].sum() if ns_coords.size else 0
        local_ew = occ[ew_coords[:, 1], ew_coords[:, 0]].sum() if ew_coords.size else 0

        ig._pressure_ns = int(local_ns - local_ew)
        ig._pressure_ew = int(local_ew - local_ns)
        state = get_rl_state(ig)
        states.append(state)

    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)

    with tf.device(tf_device):
        logits = policy_model(states_tensor)
        action_probs = tf.nn.softmax(logits).numpy()
        actions = [int(np.random.choice([0, 1], p=ap)) for ap in action_probs]

    for ig, action in zip(intersections, actions):
        ig._rl_timer += 1
        if ig._rl_timer == 1:
            ig.apply_phase(ig._rl_phase)

        min_g = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
        if action == 1 and ig._rl_timer >= min_g:
            ig._rl_phase = 1 - ig._rl_phase
            ig._rl_timer = 0

        occ = ig.city_model.occupancy_map
        ns_coords = np.array(ig.ns_in_coords)
        ew_coords = np.array(ig.ew_in_coords)
        local_ns = occ[ns_coords[:, 1], ns_coords[:, 0]].sum() if ns_coords.size else 0
        local_ew = occ[ew_coords[:, 1], ew_coords[:, 0]].sum() if ew_coords.size else 0

        reward = -(local_ns + local_ew)
        next_state = get_rl_state(ig)
        ig.memory.append((state, action, reward, next_state))

    if sum(len(ig.memory) for ig in intersections) >= 256:
        shared_memory = []
        for ig in intersections:
            shared_memory.extend(ig.memory)
            ig.memory.clear()
        train_rl_batch(policy_model, intersections[0].optimizer, shared_memory, batch_size=64)

def train_rl_batch(policy_model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, memory: list, batch_size: int = 64):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, _ = zip(*batch)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

    with tf.device(tf_device):
        with tf.GradientTape() as tape:
            logits = policy_model(states)
            neglog = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)
            loss = tf.reduce_mean(neglog * tf.stop_gradient(rewards)) - 0.01 * tf.reduce_mean(entropy)

        grads = tape.gradient(loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

def train_rl(intersection_light_group: "IntersectionLightGroup", batch_size: int = 64):
    batch = random.sample(intersection_light_group.memory, batch_size)
    states, actions, rewards, _ = zip(*batch)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

    with tf.device(tf_device):
        with tf.GradientTape() as tape:
            logits = intersection_light_group.rl_policy(states)
            neglog = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)
            loss = tf.reduce_mean(neglog * tf.stop_gradient(rewards)) - 0.01 * tf.reduce_mean(entropy)

        grads = tape.gradient(loss, intersection_light_group.rl_policy.trainable_variables)
        intersection_light_group.optimizer.apply_gradients(zip(grads, intersection_light_group.rl_policy.trainable_variables))

def warmup_simple_rl_model(model, optimizer, input_dim=13):
        dummy_input = tf.zeros((1, input_dim), dtype=tf.float32)
        _ = model(dummy_input)  # build model weights

        with tf.GradientTape() as tape:
            logits = model(dummy_input)
            loss = tf.reduce_sum(logits)  # dummy scalar

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
