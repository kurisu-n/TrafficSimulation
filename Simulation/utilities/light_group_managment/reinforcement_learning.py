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
    # 1) local queues → pressures
    ns_cells, ew_cells = intersection_light_group.get_opposite_traffic_lights().values()
    local_ns = sum(1 for tl in ns_cells for rb in tl.assigned_road_blocks if rb.occupied)
    local_ew = sum(1 for tl in ew_cells for rb in tl.assigned_road_blocks if rb.occupied)
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns

    # 2) neighbor pressures
    nbrs = list(intersection_light_group.get_neighbor_groups().values())
    sum_n_ns = sum(n._pressure_ns for n in nbrs if hasattr(n, '_pressure_ns'))
    sum_n_ew = sum(n._pressure_ew for n in nbrs if hasattr(n, '_pressure_ew'))
    cnt = max(1, len(nbrs))
    avg_n_ns = sum_n_ns / cnt
    avg_n_ew = sum_n_ew / cnt

    # 3) phase one-hot: [is_NS_green, is_EW_green]
    phase_bit = [1, 0] if intersection_light_group._rl_phase == 0 else [0, 1]

    # 4) normalized timer
    t_norm = intersection_light_group._rl_timer / getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 30)

    return [p_ns, p_ew, avg_n_ns, avg_n_ew] + phase_bit + [t_norm]


def run_neighbor_rl(intersection_light_group):
    """
    At each step, compute state, choose hold/switch, apply phase logic,
    record transition and occasionally train.
    """
    # compute & stash local pressure for neighbors
    ns_cells, ew_cells = intersection_light_group.get_opposite_traffic_lights().values()
    local_ns = sum(1 for tl in ns_cells for rb in tl.assigned_road_blocks if rb.occupied)
    local_ew = sum(1 for tl in ew_cells for rb in tl.assigned_road_blocks if rb.occupied)
    intersection_light_group._pressure_ns = local_ns - local_ew
    intersection_light_group._pressure_ew = local_ew - local_ns

    # build state and select action
    state = intersection_light_group.get_rl_state()
    logits = intersection_light_group.rl_policy(tf.constant([state], dtype=tf.float32))
    action = tf.argmax(logits[0]).numpy()  # 0=hold, 1=switch

    # apply decision: switch phase if requested and past min green
    min_g = getattr(Defaults, 'TRAFFIC_LIGHT_MIN_GREEN', 5)
    # record pre-action phase to allow transition
    prev_phase = intersection_light_group._rl_phase

    if action == 1 and intersection_light_group._rl_timer >= min_g:
        # toggle between NS(0) and EW(2)
        intersection_light_group._rl_phase = 2 if intersection_light_group._rl_phase == 0 else 0
        intersection_light_group._rl_timer = 0

    # at timer=0, apply green according to phase
    if intersection_light_group._rl_timer == 0:
        intersection_light_group._apply_phase(intersection_light_group._rl_phase)

    # advance timer
    intersection_light_group._rl_timer += 1

    # compute reward: negative sum of local queue lengths
    reward = -(local_ns + local_ew)

    # record transition
    next_state = intersection_light_group.get_rl_state()
    intersection_light_group.memory.append((state, int(action), reward, next_state))

    # periodic training
    if len(intersection_light_group.memory) >= 64:
        intersection_light_group.train_rl(batch_size=64)
        intersection_light_group.memory.clear()


def train_rl(intersection_light_group, batch_size=32):
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
        # compute policy loss weighted by reward
        neglog = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        loss = tf.reduce_mean(neglog * tf.stop_gradient(rewards))
    grads = tape.gradient(loss, intersection_light_group.rl_policy.trainable_variables)
    intersection_light_group.optimizer.apply_gradients(zip(grads, intersection_light_group.rl_policy.trainable_variables))