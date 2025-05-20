import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from Simulation.config import Defaults
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Simulation.agents.city_structure_entities.intersection_light_group import IntersectionLightGroup

# TensorFlow configuration (similar to rl_simple.py)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices and getattr(Defaults, "CUDA_GPU_ENABLED", False):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf_device = '/GPU:0'
else:
    tf_device = '/CPU:0'
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

# Hyperparameters for GAT-DQN
GAMMA = Defaults.GAT_GAMMA  # Discount factor for Q-learning
BATCH_SIZE = Defaults.GAT_BATCH_SIZE  # Mini-batch size for training
MEMORY_CAPACITY = Defaults.GAT_MEMORY_CAPACITY  # Replay memory capacity per agent
TARGET_UPDATE_EVERY = Defaults.GAT_TARGET_UPDATE_EVERY  # Frequency (in training steps) to update target network
EPS_INITIAL = Defaults.EPS_INITIAL  # Starting epsilon for ε-greedy
EPS_MIN = Defaults.EPS_MIN  # Minimum epsilon (end of decay)
EPS_DECAY_RATE = Defaults.EPS_DECAY_RATE  # Decay rate per decision step for epsilon

class GraphAttentionLayer(layers.Layer):
    """Single-head Graph Attention layer for a star graph (center node + neighbors)."""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.2)

    def build(self, input_shape):
        # input_shape[0] is (batch_size, N, F) for node features
        F = input_shape[0][-1]  # feature dimension
        # Trainable weight matrices: linear transform W and attention vector a
        self.W = self.add_weight(shape=(F, self.output_dim),
                                 initializer='glorot_uniform', trainable=True, name="W")
        self.a = self.add_weight(shape=(2 * self.output_dim, 1),
                                 initializer='glorot_uniform', trainable=True, name="attn_weight")
        super().build(input_shape)

    def call(self, inputs):
        """Compute attention-weighted sum of neighbor features for the center node."""
        node_features, mask = inputs  # node_features: (batch, N, F), mask: (batch, N)
        # Apply linear transformation W to all node features
        Wf = tf.tensordot(node_features, self.W, axes=[[2], [0]])  # shape: (batch, N, output_dim)
        # Extract the center node's transformed feature (index 0)
        center_feature = Wf[:, 0:1, :]  # shape: (batch, 1, output_dim)
        # Repeat center feature alongside each neighbor to concatenate for attention scores
        # center_feature_broadcast: (batch, N, output_dim)
        center_feature_broadcast = tf.broadcast_to(center_feature, tf.shape(Wf))
        # Concatenate center and neighbor features for attention score computation
        concat_features = tf.concat([center_feature_broadcast, Wf], axis=-1)  # (batch, N, 2*output_dim)
        # Compute attention logits e_ij for each node j (including j=0 self) using attention weights a
        e = tf.tensordot(concat_features, self.a, axes=1)  # shape: (batch, N, 1)
        e = tf.squeeze(e, axis=-1)  # shape: (batch, N)
        e = self.leaky_relu(e)  # LeakyReLU activation on logits
        # Mask out nonexistent neighbors by setting their logits to a large negative value
        if mask is not None:
            mask = tf.cast(mask, tf.float32)  # (batch, N), 1 for real node, 0 for padded node
            e = e + (1.0 - mask) * -1e9  # -inf for padded nodes so they get 0 attention weight
        # Normalize attention coefficients across all neighbors (and self) for each sample
        alpha = tf.nn.softmax(e, axis=1)  # shape: (batch, N)
        alpha = tf.expand_dims(alpha, axis=-1)  # shape: (batch, N, 1)
        # Compute attention-weighted sum of **transformed** neighbor features (including self)
        # This yields the aggregated representation for the center node
        center_agg = tf.reduce_sum(alpha * Wf, axis=1)  # shape: (batch, output_dim)
        return center_agg


def make_gat_dqn_net(node_feature_dim=9, max_neighbors=4, gat_output_dim=16, hidden_units=32, num_actions=2):
    """
    Construct the GAT-DQN neural network:
      - Inputs: node features for center + neighbors, and a mask for valid neighbors.
      - Graph Attention layer to aggregate neighbor info.
      - A fully-connected feedforward network to output Q-values for each action.
    """
    with tf.device(tf_device):
        # Define two inputs: node features (including center and neighbor nodes) and neighbor mask
        N = 1 + max_neighbors  # total nodes = 1 center + max_neighbors
        node_features_input = tf.keras.Input(shape=(N, node_feature_dim), name="node_features")
        mask_input = tf.keras.Input(shape=(N,), name="neighbor_mask")
        # Graph Attention layer (attention over center and neighbor nodes)
        gat_layer = GraphAttentionLayer(output_dim=gat_output_dim, name="graph_attention")
        center_rep = gat_layer([node_features_input, mask_input])  # shape: (None, gat_output_dim)
        # Optionally, include a non-linear activation (ReLU) on the GAT output
        center_rep = layers.ReLU()(center_rep)
        # Append a fully-connected network for Q-value prediction
        x = layers.Dense(hidden_units, activation='relu')(center_rep)
        x = layers.Dense(hidden_units, activation='relu')(x)
        q_values = layers.Dense(num_actions, name="q_values")(x)  # output Q for each action
        model = tf.keras.Model(inputs=[node_features_input, mask_input], outputs=q_values)
    return model


def get_gat_state(intersection_light_group: "IntersectionLightGroup"):
    """
    Compute graph-based state representation for a given intersection:
    Returns (node_features, mask) where:
      - node_features is a NumPy array of shape (5, feature_dim) including the center node (index 0) and up to 4 neighbors.
      - mask is a NumPy array of shape (5,) indicating valid nodes (1 for real neighbor, 0 for padding).
    """
    city = intersection_light_group.city_model
    occ_map = city.occupancy_map  # current occupancy grid (NumPy array)
    # Center intersection features
    ns_coords = intersection_light_group.ns_in_coords
    ew_coords = intersection_light_group.ew_in_coords
    local_ns = occ_map[ns_coords[:, 1], ns_coords[:, 0]].sum() if ns_coords.size else 0
    local_ew = occ_map[ew_coords[:, 1], ew_coords[:, 0]].sum() if ew_coords.size else 0
    p_ns = local_ns - local_ew
    p_ew = local_ew - local_ns
    # Current phase (0 or 1) and normalized time elapsed in that phase
    phase = getattr(intersection_light_group, '_rl_phase', 0)
    phase_bit = [1.0, 0.0] if phase == 0 else [0.0, 1.0]
    t_norm = intersection_light_group._rl_timer / getattr(Defaults, "TRAFFIC_LIGHT_MAX_GREEN", 30)
    # Static intersection features
    intersection_size = intersection_light_group.intersection_size
    penalty_score = intersection_light_group.penalty_score
    center_features = [
        float(local_ns), float(local_ew),
        float(p_ns), float(p_ew),
        phase_bit[0], phase_bit[1],
        float(t_norm),
        float(intersection_size), float(penalty_score)
    ]
    # Neighbor nodes features (up to 4 neighbors: N, S, E, W)
    neighbor_features = []
    neighbor_mask = []
    neighbors = intersection_light_group.get_neighbor_groups()
    # Define a fixed order for neighbors to maintain consistency (N, S, E, W)
    directions = ["N", "S", "E", "W"]
    for d in directions:
        if d in neighbors:
            ng = neighbors[d]
            # Compute neighbor's local queue lengths from occupancy map
            ns_coords_ng = ng.ns_in_coords
            ew_coords_ng = ng.ew_in_coords
            local_ns_ng = occ_map[ns_coords_ng[:, 1], ns_coords_ng[:, 0]].sum() if ns_coords_ng.size else 0
            local_ew_ng = occ_map[ew_coords_ng[:, 1], ew_coords_ng[:, 0]].sum() if ew_coords_ng.size else 0
            p_ns_ng = local_ns_ng - local_ew_ng
            p_ew_ng = local_ew_ng - local_ns_ng
            # Neighbor's current phase and timer
            phase_ng = getattr(ng, '_rl_phase', 0)
            phase_bit_ng = [1.0, 0.0] if phase_ng == 0 else [0.0, 1.0]
            t_norm_ng = getattr(ng, '_rl_timer', 0) / getattr(Defaults, "TRAFFIC_LIGHT_MAX_GREEN", 30)
            # Static features of neighbor
            intersection_size_ng = getattr(ng, 'intersection_size', 0.0)
            penalty_ng = getattr(ng, 'penalty_score', 0.0)
            neighbor_features.append([
                float(local_ns_ng), float(local_ew_ng),
                float(p_ns_ng), float(p_ew_ng),
                phase_bit_ng[0], phase_bit_ng[1],
                float(t_norm_ng),
                float(intersection_size_ng), float(penalty_ng)
            ])
            neighbor_mask.append(1.0)
        else:
            # No neighbor in this direction: use dummy features and mask 0
            neighbor_features.append([0.0] * len(center_features))
            neighbor_mask.append(0.0)
    # Combine center and neighbor features into one array
    node_features = np.vstack([center_features, np.array(neighbor_features, dtype=float)])
    mask = np.array([1.0] + neighbor_mask, dtype=float)  # mask[0]=1 for center, neighbor_mask for others
    return node_features, mask


def run_gat_dqn_control(intersection_light_group: "IntersectionLightGroup"):
    """
    Execute one decision step for a single intersection (single-agent mode).
    Chooses an action via ε-greedy policy, applies the action, and stores the experience.
    Also triggers training and target network updates when enough experiences have been collected.
    """
    # Initialize RL phase/timer if not already set
    if not hasattr(intersection_light_group, '_rl_phase'):
        intersection_light_group._rl_phase = 0
        intersection_light_group._rl_timer = 0
    # Compute current state (graph features + mask)
    node_features, mask = get_gat_state(intersection_light_group)
    state = (node_features, mask)
    state_tensor = tf.convert_to_tensor(node_features[None, ...], dtype=tf.float32)  # shape (1,5,9)
    mask_tensor = tf.convert_to_tensor(mask[None, ...], dtype=tf.float32)  # shape (1,5)
    # Forward pass: get Q-values for current state
    with tf.device(tf_device):
        q_values = intersection_light_group.gat_q_func(state_tensor, mask_tensor)[0].numpy()  # shape (2,)
    # Epsilon-greedy action selection
    epsilon = intersection_light_group.epsilon
    if random.random() < epsilon:
        # Explore: random action
        action = random.randrange(len(q_values))
    else:
        # Exploit: choose best action
        action = int(np.argmax(q_values))
    # Decay epsilon (for future steps)
    intersection_light_group.epsilon = max(EPS_MIN, intersection_light_group.epsilon - EPS_DECAY_RATE)
    # Apply the chosen action (traffic light phase control)
    intersection_light_group._rl_timer += 1
    if intersection_light_group._rl_timer == 1:
        # On first tick of a phase (timer reset), actually apply the current phase setting to lights
        intersection_light_group.apply_phase(intersection_light_group._rl_phase)
    # If action=1 (switch) and minimum green time has elapsed, toggle the phase
    min_green = getattr(Defaults, "TRAFFIC_LIGHT_MIN_GREEN", 5)
    if action == 1 and intersection_light_group._rl_timer >= min_green:
        # Switch phase (0->1 or 1->0)
        intersection_light_group._rl_phase = 1 - intersection_light_group._rl_phase
        intersection_light_group._rl_timer = 0  # reset timer after switching
    # Compute reward for this action
    # Reward includes local queue penalty and global metrics (average trip duration & time per block)
    occ_map = intersection_light_group.city_model.occupancy_map
    # Local queue lengths after action (for immediate reward, use current state occupancies)
    local_ns = state[0][0, 0] if state[0].shape[0] > 0 else 0  # using center_features from state (index 0)
    local_ew = state[0][0, 1] if state[0].shape[0] > 0 else 0
    local_queue_penalty = local_ns + local_ew  # total cars waiting at this intersection
    # Global metrics from dynamic traffic generator
    dta = getattr(intersection_light_group.city_model, "dynamic_traffic_generator", None)
    if dta and hasattr(dta, "cached_stats"):
        # Use average trip duration and average time per block (per unit distance) from cached_stats
        avg_trip_duration = 0.0
        avg_time_per_block = 0.0
        # Combine internal and through trip metrics for a global average (if available)
        if "avg_duration_internal" in dta.cached_stats and "avg_duration_through" in dta.cached_stats:
            avg_trip_duration = (dta.cached_stats["avg_duration_internal"] + dta.cached_stats[
                "avg_duration_through"]) / 2.0
        if "avg_time_per_unit_internal" in dta.cached_stats and "avg_time_per_unit_through" in dta.cached_stats:
            avg_time_per_block = (dta.cached_stats["avg_time_per_unit_internal"] + dta.cached_stats[
                "avg_time_per_unit_through"]) / 2.0
        # Negative reward (penalty) for longer trip times
        global_time_penalty = 0.01 * avg_trip_duration + 1.0 * avg_time_per_block
    else:
        # If global metrics not available, fallback to using only local queue
        global_time_penalty = 0.0
    # Define reward (negative of penalties so that lower queues/times yield higher reward)
    reward = - (float(local_queue_penalty) + float(global_time_penalty))
    # Get next state after action (for experience storage)
    next_node_features, next_mask = get_gat_state(intersection_light_group)
    next_state = (next_node_features, next_mask)
    # Store experience in replay buffer
    intersection_light_group.memory.append((state[0], state[1], action, reward, next_state[0], next_state[1]))
    # Train the DQN when enough samples are in memory
    if len(intersection_light_group.memory) >= BATCH_SIZE:
        _train_dqn(intersection_light_group)
        # Periodically update target network to track policy network weights
        if intersection_light_group.train_step_count % TARGET_UPDATE_EVERY == 0:
            try:
                intersection_light_group.target_policy.set_weights(intersection_light_group.rl_policy.get_weights())
            except Exception:
                # In case the model isn't built, skip (will build on next use)
                pass


def run_batched_gat_dqn_control(intersections: list["IntersectionLightGroup"]):
    """
    Batched GAT-DQN control:
     1) Vectorise feature assembly into a single big tensor
     2) Use the compiled _tf_q_values() for each agent
     3) Store each agent's transition
     4) Call _train_dqn() (which itself uses _tf_dqn_train)
     5) Sync target network periodically
    """
    N = len(intersections)
    # 1) Pre-allocate buffers and build them in one Python loop
    feat_buf = np.empty((N, 1 + len(Defaults.AVAILABLE_DIRECTIONS), 9), dtype=np.float32)
    mask_buf = np.empty((N, 1 + len(Defaults.AVAILABLE_DIRECTIONS)), dtype=np.float32)

    for i, ig in enumerate(intersections):
        feat, mask = get_gat_state(ig)
        feat_buf[i] = feat
        mask_buf[i] = mask

    # Single conversion from NumPy → Tensor
    feat_tensor = tf.convert_to_tensor(feat_buf)  # shape (N, 5, 9)
    mask_tensor = tf.convert_to_tensor(mask_buf)  # shape (N, 5)

    # 2) Decision & experience collection
    for i, ig in enumerate(intersections):
        # --- ε-greedy action via compiled forward pass ---
        state_feat = feat_tensor[i : i + 1]    # shape (1,5,9)
        state_mask = mask_tensor[i : i + 1]    # shape (1,5)
        q_vals = _tf_q_values(ig.rl_policy, state_feat, state_mask)[0].numpy()

        if random.random() < ig.epsilon:
            action = random.randrange(len(q_vals))
        else:
            action = int(np.argmax(q_vals))

        # decay epsilon
        ig.epsilon = max(EPS_MIN, ig.epsilon - EPS_DECAY_RATE)

        # --- apply traffic-light action ---
        ig._rl_timer += 1
        if ig._rl_timer == 1:
            ig.apply_phase(ig._rl_phase)
        if action == 1 and ig._rl_timer >= Defaults.GAT_TRAFFIC_RL_MIN_GREEN:
            ig._rl_phase = 1 - ig._rl_phase
            ig._rl_timer = 0

        # --- compute reward: local queue + global metrics ---
        occ = ig.city_model.occupancy_map
        ns = occ[ig.ns_in_coords[:, 1], ig.ns_in_coords[:, 0]].sum() if ig.ns_in_coords.size else 0
        ew = occ[ig.ew_in_coords[:, 1], ig.ew_in_coords[:, 0]].sum() if ig.ew_in_coords.size else 0
        local_penalty = ns + ew

        dta = getattr(ig.city_model, "dynamic_traffic_generator", None)
        global_penalty = 0.0
        if dta and hasattr(dta, "cached_stats"):
            avg_dur = 0.5 * (dta.cached_stats.get("avg_duration_internal", 0.0)
                            + dta.cached_stats.get("avg_duration_through", 0.0))
            avg_tpb = 0.5 * (dta.cached_stats.get("avg_time_per_unit_internal", 0.0)
                            + dta.cached_stats.get("avg_time_per_unit_through", 0.0))
            global_penalty = 0.01 * avg_dur + 1.0 * avg_tpb

        reward = - (float(local_penalty) + float(global_penalty))

        # --- store transition in this agent's replay buffer ---
        next_feat, next_mask = get_gat_state(ig)
        ig.memory.append((
            feat_buf[i],      # state features
            mask_buf[i],      # state mask
            action,           # action taken
            reward,           # reward received
            next_feat,        # next state features
            next_mask         # next state mask
        ))

    # 3) Train each agent if it has enough samples & periodically sync target
    for ig in intersections:
        if len(ig.memory) >= BATCH_SIZE:
            _train_dqn(ig)  # uses the compiled @_tf_dqn_train internally
            if ig.train_step_count % TARGET_UPDATE_EVERY == 0:
                ig.target_policy.set_weights(ig.rl_policy.get_weights())


@tf.function(jit_compile=True)
def _tf_q_values(model, feat, mask):
    return model([feat, mask])

@tf.function(jit_compile=True)
def _tf_dqn_train(model, target, opt,
                  s_feat, s_mask,
                  actions, rewards,
                  ns_feat, ns_mask):
    with tf.GradientTape() as tape:
        q_pred     = model([s_feat,  s_mask])
        act_onehot = tf.one_hot(actions, depth=q_pred.shape[-1], dtype=tf.float32)
        q_pred_sa  = tf.reduce_sum(q_pred * act_onehot, axis=1)

        q_next     = target([ns_feat, ns_mask])
        q_next_max = tf.reduce_max(q_next, axis=1)
        td_target  = rewards + GAMMA * q_next_max
        loss       = tf.reduce_mean(tf.square(q_pred_sa - tf.stop_gradient(td_target)))

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

def _train_dqn(intersection_light_group: "IntersectionLightGroup"):
    """Internal helper to train the DQN for a given intersection from its replay memory."""
    ig = intersection_light_group
    # Increment training step counter
    if not hasattr(ig, "train_step_count"):
        ig.train_step_count = 0
    ig.train_step_count += 1
    # Sample a random mini-batch from memory
    batch = random.sample(ig.memory, BATCH_SIZE)
    states_feat, states_mask, actions, rewards, next_states_feat, next_states_mask = zip(*batch)
    states_feat = tf.convert_to_tensor(states_feat, dtype=tf.float32)  # shape (BATCH_SIZE, 5, 9)
    states_mask = tf.convert_to_tensor(states_mask, dtype=tf.float32)  # shape (BATCH_SIZE, 5)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # shape (BATCH_SIZE,)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)  # shape (BATCH_SIZE,)
    next_states_feat = tf.convert_to_tensor(next_states_feat, dtype=tf.float32)  # shape (BATCH_SIZE, 5, 9)
    next_states_mask = tf.convert_to_tensor(next_states_mask, dtype=tf.float32)  # shape (BATCH_SIZE, 5)
    # Perform one-step DQN update using the target network for next state estimates

    if len(ig.memory) >= BATCH_SIZE:
        # unpack your batch into tf.Tensor tensors: s_f, s_m, actions, rewards, ns_f, ns_m
        intersection_light_group.gat_train_func(states_feat, states_mask, actions, rewards, next_states_feat, next_states_mask)
        if ig.train_step_count % TARGET_UPDATE_EVERY == 0:
            ig.target_policy.set_weights(ig.rl_policy.get_weights())

