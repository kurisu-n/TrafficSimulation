import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from Simulation.config import Defaults
from typing import TYPE_CHECKING, List, Tuple

from Simulation.utilities.numba_utilities import compute_cross_pressure, compute_local_and_cross_pressure, \
    avg_pressures_in_neighbors

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

def compute_pressure(
    ig: "IntersectionLightGroup",
    flow_map: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute NS and EW pressures for a given light-group using cached coords.
    Handles empty coord lists by reshaping to (0, 2).
    Returns: (local_ns, local_ew, p_ns, p_ew)
    """
    # Cache and normalize coordinate grids on first use
    if not hasattr(ig, 'ns_coords'):
        ig.ns_coords = np.array(ig.ns_in_coords, dtype=np.int64).reshape(-1, 2)
        ig.ew_coords = np.array(ig.ew_in_coords, dtype=np.int64).reshape(-1, 2)
    # Ensure shape correctness in case coords were modified
    if ig.ns_coords.ndim == 1:
        ig.ns_coords = ig.ns_coords.reshape(-1, 2)
    if ig.ew_coords.ndim == 1:
        ig.ew_coords = ig.ew_coords.reshape(-1, 2)

    # Sum flows; if no coords, sum over empty returns 0.0
    local_ns = float(flow_map[ig.ns_coords[:, 1], ig.ns_coords[:, 0]].sum())
    local_ew = float(flow_map[ig.ew_coords[:, 1], ig.ew_coords[:, 0]].sum())

    # Pressure definitions
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    # Store for neighbor averaging and later use
    ig.pressure_ns = p_ns
    ig.pressure_ew = p_ew
    return local_ns, local_ew, p_ns, p_ew


def avg_neighbor_pressures(
    nbrs: List["IntersectionLightGroup"],
    flow_map: np.ndarray
) -> Tuple[float, float]:
    """
    Ensure each neighbor has up-to-date _pressure_*, then return simple mean.
    """
    for n in nbrs:
        if not hasattr(n, 'pressure_ns'):
            compute_pressure(n, flow_map)

    ps_ns = np.fromiter((n.pressure_ns for n in nbrs), dtype=np.float64)
    ps_ew = np.fromiter((n.pressure_ew for n in nbrs), dtype=np.float64)

    cnt = max(1, ps_ns.size)
    return ps_ns.sum() / cnt, ps_ew.sum() / cnt



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

def get_rl_state(
    intersection_light_group: "IntersectionLightGroup"
) -> List[float]:
    city = intersection_light_group.city_model
    occupancy = city.occupancy_map
    stuck_map = city.stuck_map

    local_ns, local_ew, p_ns, p_ew = compute_pressure(
        intersection_light_group, occupancy
    )

    nbrs = list(intersection_light_group.get_neighbor_groups().values())
    nbrs_cnt = max(1, len(nbrs))

    phase_bit = [1.0, 0.0] if intersection_light_group._rl_phase == 0 else [0.0, 1.0]
    timer_norm = (
        intersection_light_group.rl_timer /
        float(getattr(intersection_light_group.__class__, 'TRAFFIC_LIGHT_MAX_GREEN', 30))
    )

    state = [
        local_ns, local_ew,
        p_ns, p_ew,
        *phase_bit,
        timer_norm
    ]

    if Defaults.SRL_INPUT_DIMENSIONS > 7:
        size = intersection_light_group.intersection_size
        penalty = intersection_light_group.penalty_score
        avg_size = sum(getattr(n, 'intersection_size', 0) for n in nbrs) / nbrs_cnt
        avg_pen = sum(getattr(n, 'penalty_score', 0)   for n in nbrs) / nbrs_cnt
        state += [size, penalty, avg_size, avg_pen]

    if Defaults.SRL_INPUT_DIMENSIONS > 11:
        avg_n_ns, avg_n_ew = avg_neighbor_pressures(nbrs, occupancy)
        state += [avg_n_ns, avg_n_ew]

    if Defaults.SRL_INPUT_DIMENSIONS > 13:
        ln_s, le_s, ps_ns, ps_ew = compute_pressure(
            intersection_light_group, stuck_map
        )
        state += [ln_s, le_s, ps_ns, ps_ew]

    if Defaults.SRL_INPUT_DIMENSIONS > 17:
        avg_sn_ns, avg_sn_ew = avg_neighbor_pressures(nbrs, stuck_map)
        state += [avg_sn_ns, avg_sn_ew]

    return state

# -----------------------------------------------------------------------------
# Optimized RL control loop
# -----------------------------------------------------------------------------

def run_rl_control(
    intersection_light_group: "IntersectionLightGroup"
) -> None:
    # Initialize phase and timer
    if not hasattr(intersection_light_group, '_rl_phase'):
        intersection_light_group._rl_phase = 0
        intersection_light_group.rl_timer = 0

    # Compute and store current pressures
    city = intersection_light_group.city_model
    compute_pressure(intersection_light_group, city.occupancy_map)
    intersection_light_group.pressure_ns = int(intersection_light_group.pressure_ns)
    intersection_light_group.pressure_ew = int(intersection_light_group.pressure_ew)

    # Create RL state & get action
    state = get_rl_state(intersection_light_group)
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    with tf.device(tf_device):
        logits = intersection_light_group.rl_policy(state_tensor)
        action_probs = tf.nn.softmax(logits[0])
        action = int(np.random.choice([0, 1], p=action_probs.numpy()))

    # Apply phase on first timestep
    intersection_light_group.rl_timer += 1
    if intersection_light_group.rl_timer == 1:
        intersection_light_group.apply_phase(intersection_light_group._rl_phase)

    # Phase switch logic
    if action == 1 and intersection_light_group.rl_timer >= Defaults.SRL_MIN_GREEN:
        intersection_light_group._rl_phase = 1 - intersection_light_group._rl_phase
        intersection_light_group.rl_timer = 0

    # Reward computation
    neg_reward = intersection_light_group.pressure_ns + intersection_light_group.pressure_ew
    if Defaults.SRL_PUNISH_STUCK:
        compute_pressure(intersection_light_group, city.stuck_map)
        neg_reward += (
            intersection_light_group.pressure_ns +
            intersection_light_group.pressure_ew
        ) * Defaults.SRL_PUNISH_STUCK_FACTOR
    reward = -neg_reward

    # Store transition
    next_state = get_rl_state(intersection_light_group)
    intersection_light_group.memory.append((state, action, reward, next_state))

    # Periodic training
    if len(intersection_light_group.memory) >= UPDATE_EVERY:
        train_rl(intersection_light_group, BATCH_SIZE)
        intersection_light_group.memory.clear()

# -----------------------------------------------------------------------------
# Optimized batched RL control
# -----------------------------------------------------------------------------

def run_batched_rl_control(
    intersections: List["IntersectionLightGroup"],
    policy_model: tf.keras.Model
) -> None:
    # Precompute pressures and states
    states: List[List[float]] = []
    for ig in intersections:
        city = ig.city_model
        compute_pressure(ig, city.occupancy_map)
        ig.pressure_ns = ig.pressure_ns
        ig.pressure_ew = ig.pressure_ew
        states.append(get_rl_state(ig))

    # Batch inference
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    with tf.device(tf_device):
        logits = policy_model(states_tensor)
        action_probs = tf.nn.softmax(logits, axis=1).numpy()
        actions = [int(np.random.choice([0, 1], p=ap)) for ap in action_probs]

    # Apply actions and collect transitions
    for ig, action in zip(intersections, actions):
        # Phase timing
        if not hasattr(ig, '_rl_phase'):
            ig._rl_phase = 0
            ig.rl_timer = 0
        ig.rl_timer += 1
        if ig.rl_timer == 1:
            ig.apply_phase(ig._rl_phase)
        if action == 1 and ig.rl_timer >= Defaults.SRL_MIN_GREEN:
            ig._rl_phase = 1 - ig._rl_phase
            ig.rl_timer = 0

        # Compute reward
        city = ig.city_model
        neg_reward = ig.pressure_ns + ig.pressure_ew
        if Defaults.SRL_INPUT_DIMENSIONS > 11 and Defaults.SRL_PUNISH_STUCK:
            compute_pressure(ig, city.stuck_map)
            neg_reward += (ig.pressure_ns + ig.pressure_ew) * Defaults.SRL_PUNISH_STUCK_FACTOR
        if Defaults.SRL_INPUT_DIMENSIONS > 15 and Defaults.SRL_PUNISH_NEIGHBOR:
            nbrs = list(ig.get_neighbor_groups().values())
            avg_n_ns_stuck, avg_n_ew_stuck = avg_neighbor_pressures(nbrs, city.stuck_map)
            neg_reward += (avg_n_ns_stuck + avg_n_ew_stuck) * Defaults.SRL_PUNISH_NEIGHBOR_FACTOR
        reward = -neg_reward

        # Store transition
        next_state = get_rl_state(ig)
        ig.memory.append((states[intersections.index(ig)], action, reward, next_state))

    # Batch training
    total_mem = sum(len(ig.memory) for ig in intersections)
    if total_mem >= UPDATE_EVERY:
        shared = []
        for ig in intersections:
            shared.extend(ig.memory)
            ig.memory.clear()
        train_rl_batch(policy_model, intersections[0].optimizer, shared, batch_size=64)



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
