import random
from typing import TYPE_CHECKING
from Simulation.config import Defaults
import tensorflow as tf

if TYPE_CHECKING:
    from Simulation.agents.city_structure_entities.intersection_light_group import IntersectionLightGroup


def make_policy_net(input_dim, hidden=16):
    """
    Build a small MLP policy: 2 hidden layers, outputs logits for 2 actions.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden, activation='relu')(inputs)
    x = tf.keras.layers.Dense(hidden, activation='relu')(x)
    logits = tf.keras.layers.Dense(2)(x)  # actions: 0=hold, 1=switch
    return tf.keras.Model(inputs, logits)


def get_rl_state(intersection_light_group: "IntersectionLightGroup") -> list[float]:
    """
    Compose the state vector: [p_ns, p_ew, avg_nbr_p_ns, avg_nbr_p_ew, phase_bit, t_norm]
    """
    city = intersection_light_group.city_model
    occupancy = city.occupancy_map

    # 1) local queues → pressures
    ns_cells, ew_cells = intersection_light_group.get_opposite_traffic_lights().values()
    local_ns = sum(
        1 for tl in ns_cells for rb in tl.assigned_road_blocks
        if occupancy[rb.position[1], rb.position[0]] == 1
    )
    local_ew = sum(
        1 for tl in ew_cells for rb in tl.assigned_road_blocks
        if occupancy[rb.position[1], rb.position[0]] == 1
    )
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    # 2) neighbor pressures (from last RL update)
    nbrs = list(intersection_light_group.get_neighbor_groups().values())
    sum_n_ns = sum(n._pressure_ns for n in nbrs if hasattr(n, '_pressure_ns'))
    sum_n_ew = sum(n._pressure_ew for n in nbrs if hasattr(n, '_pressure_ew'))
    cnt = max(1, len(nbrs))
    avg_n_ns = sum_n_ns / cnt
    avg_n_ew = sum_n_ew / cnt

    # 3) phase one-hot using the RL pointer
    phase_bit = [1, 0] if intersection_light_group._rl_phase == 0 else [0, 1]

    # 4) normalized timer
    t_norm = intersection_light_group._rl_timer / getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 30)

    return [p_ns, p_ew, avg_n_ns, avg_n_ew] + phase_bit + [t_norm]


def run_rl_control(intersection_light_group: "IntersectionLightGroup"):
    """
    Reinforcement-learning-based control:
    - Computes local pressure for neighbor coordination
    - Chooses to hold or switch based on policy net
    - Uses apply_phase() + global transition logic
    """
    # Initialize RL phase pointer and timer
    if not hasattr(intersection_light_group, '_rl_phase'):
        intersection_light_group._rl_phase = 0  # 0 = NS green, 1 = EW green
        intersection_light_group._rl_timer = 0

    # 1) Compute & stash local pressure for neighbor influence
    city = intersection_light_group.city_model
    occupancy = city.occupancy_map
    ns_cells, ew_cells = intersection_light_group.get_opposite_traffic_lights().values()
    local_ns = sum(
        1 for tl in ns_cells for rb in tl.assigned_road_blocks
        if occupancy[rb.position[1], rb.position[0]] == 1
    )
    local_ew = sum(
        1 for tl in ew_cells for rb in tl.assigned_road_blocks
        if occupancy[rb.position[1], rb.position[0]] == 1
    )
    intersection_light_group._pressure_ns = local_ns - local_ew
    intersection_light_group._pressure_ew = local_ew - local_ns

    # 2) Build state & select action
    state = get_rl_state(intersection_light_group)
    logits = intersection_light_group.rl_policy(tf.constant([state], dtype=tf.float32))
    action = tf.argmax(logits[0]).numpy()  # 0 = hold, 1 = switch

    # 3) Apply-phase logic
    intersection_light_group._rl_timer += 1
    # On first tick of a phase, request green via apply_phase
    if intersection_light_group._rl_timer == 1:
        intersection_light_group.apply_phase(intersection_light_group._rl_phase)

    # If policy says switch and min green elapsed, toggle pointer & reset
    min_g = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
    if action == 1 and intersection_light_group._rl_timer >= min_g:
        intersection_light_group._rl_phase = 1 - intersection_light_group._rl_phase
        intersection_light_group._rl_timer = 0

    # 4) Record transition & train
    reward = -(local_ns + local_ew)
    next_state = get_rl_state(intersection_light_group)
    intersection_light_group.memory.append((state, int(action), reward, next_state))

    if len(intersection_light_group.memory) >= 64:
        train_rl(intersection_light_group, batch_size=64)
        intersection_light_group.memory.clear()


def train_rl(intersection_light_group: "IntersectionLightGroup", batch_size: int = 32):
    """
    Simple policy-gradient update: treats reward as advantage.
    """
    batch = random.sample(intersection_light_group.memory, batch_size)
    states, actions, rewards, _ = zip(*batch)
    states = tf.constant(states, dtype=tf.float32)
    actions = tf.constant(actions, dtype=tf.int32)
    rewards = tf.constant(rewards, dtype=tf.float32)

    with tf.GradientTape() as tape:
        logits = intersection_light_group.rl_policy(states)
        neglog = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        loss = tf.reduce_mean(neglog * tf.stop_gradient(rewards))
    grads = tape.gradient(loss, intersection_light_group.rl_policy.trainable_variables)
    intersection_light_group.optimizer.apply_gradients(zip(grads, intersection_light_group.rl_policy.trainable_variables))
