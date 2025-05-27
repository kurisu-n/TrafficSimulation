# rl_a2c.py
import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, optimizers
from Simulation.config import Defaults

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices and Defaults.CUDA_GPU_ENABLED:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf_device = '/GPU:0'
else:
    tf_device = '/CPU:0'
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

HIDDEN_LAYERS = Defaults.A2C_HIDDEN_LAYERS
HIDDEN_LAYER_SIZE = Defaults.A2C_HIDDEN_LAYER_SIZE

################################################################################
#  NETWORK FACTORIES
################################################################################
def make_actor_critic(input_dim=13, hidden=HIDDEN_LAYER_SIZE):
    inputs = tf.keras.Input(shape=(input_dim,), name="state")
    x = layers.Dense(hidden, activation='relu')(inputs)

    for _ in range(HIDDEN_LAYERS - 1):
        x = layers.Dense(hidden, activation='relu')(x)

    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    logits   = layers.Dense(2, name="policy_logits")(x)       # actor
    value    = layers.Dense(1, name="state_value")(x)         # critic

    actor  = tf.keras.Model(inputs, logits)
    critic = tf.keras.Model(inputs, value)
    return actor, critic

################################################################################
#  STATE ENCODING     (unchanged – copied verbatim)
################################################################################
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
    t_norm = intersection_light_group.rl_timer / getattr(Defaults, 'TRAFFIC_LIGHT_MAX_GREEN', 30)

    intersection_size_factor = intersection_light_group.intersection_size
    penalty_score = intersection_light_group.penalty_score

    avg_penalty_intersection_size = sum(getattr(n, 'intersection_size_factor', 0) for n in nbrs)/cnt
    avg_penalty_score = sum(getattr(n, 'penalty_score', 0) for n in nbrs)/cnt

    return [float(local_ns), float(local_ew), p_ns, p_ew, avg_n_ns, avg_n_ew] + phase_bit + [t_norm] + [intersection_size_factor,penalty_score,avg_penalty_intersection_size, avg_penalty_score]
################################################################################
#  A2C HYPER-PARAMETERS
################################################################################
GAMMA       = Defaults.A2C_GAMMA          # discount
LAMBDA      = Defaults.A2C_LAMBDA           # GAE(λ)
ROLL_OUT    = Defaults.A2C_UPDATE_EVERY             # env steps before an update
BATCH_SIZE  = Defaults.A2C_BATCH_SIZE            # SGD minibatch
ENTROPY_MAX = Defaults.A2C_ENTROPY_MAX
ENTROPY_MIN = Defaults.A2C_ENTROPY_MIN
ENTROPY_DECAY_STEPS = Defaults.A2C_ENTROPY_DECAY_STEPS

################################################################################
#  TRAJECTORY BUFFER (shared across all intersections)
################################################################################
class TrajectoryBuffer:
    __slots__ = ("s","a","r","v","logp")
    def __init__(self):
        self.clear()
    def store(self, s,a,r,v,logp):
        self.s.append(s); self.a.append(a); self.r.append(r); self.v.append(v); self.logp.append(logp)
    def clear(self):
        self.s, self.a, self.r, self.v, self.logp = [],[],[],[],[]
    def size(self): return len(self.a)

BUFFER = TrajectoryBuffer()      # one global rollout buffer
GLOBAL_STEP = tf.Variable(0, dtype=tf.int64)

################################################################################
#  MAIN CONTROL (called once per sim-step from run_batched_rl_control)
################################################################################
def run_a2c_control(
    intersections: list,
    actor: tf.keras.Model,
    critic: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
):
    """
    • Gathers the 9-D state from every IntersectionLightGroup.
    • Samples an action (keep ↔ switch phase) from the shared actor.
    • Applies the action + environment rules, computes scalar reward.
    • Stores (s, a, r, v, logπ(a|s)) in the global rollout buffer.
    • When ROLL_OUT transitions collected → one A2C update step.
    """

    # ------------------------------------------------------------------ #
    # 1) build batch of states and get policy/value
    # ------------------------------------------------------------------ #
    states_py = [get_rl_state(ig) for ig in intersections]
    s_batch   = tf.convert_to_tensor(states_py, dtype=tf.float32)       # (B,9)

    logits    = actor(s_batch)               # (B,2)
    actions   = tf.squeeze(
        tf.random.categorical(logits, 1),    # sample from π
        axis=1
    )                                         # (B,)

    logp_batch  = tf.nn.log_softmax(logits)   # (B,2)
    value_batch = tf.squeeze(critic(s_batch), axis=1)  # (B,)

    # ------------------------------------------------------------------ #
    # 2) step every intersection with its chosen action
    # ------------------------------------------------------------------ #
    rewards = []

    for idx, (ig, act) in enumerate(zip(intersections, actions.numpy())):
        # -- timer & min-green bookkeeping (identical to old logic) ----
        ig.rl_timer += 1
        if ig.rl_timer == 1:
            ig.apply_phase(ig._rl_phase)

        if (
            act == 1
            and ig.rl_timer >= getattr(Defaults, "TRAFFIC_LIGHT_MIN_GREEN", 5)
        ):
            ig._rl_phase = 1 - ig._rl_phase   # toggle phase
            ig.rl_timer = 0

        # -- compute reward -------------------------------------------
        occ   = ig.city_model.occupancy_map
        ns_q  = (
            occ[ig.ns_in_coords[:, 1], ig.ns_in_coords[:, 0]].sum()
            if ig.ns_in_coords.size
            else 0
        )
        ew_q  = (
            occ[ig.ew_in_coords[:, 1], ig.ew_in_coords[:, 0]].sum()
            if ig.ew_in_coords.size
            else 0
        )

        queue_len = ns_q + ew_q
        pressure  = (ns_q - ew_q) ** 2
        r_t       = -float(queue_len + 0.25 * pressure)
        rewards.append(r_t)

        # -- log-prob and V(s) for this sample ------------------------
        logp = tf.gather(logp_batch[idx], act)      # scalar
        v_t  = value_batch[idx]                     # scalar

        # -- store transition in shared rollout buffer ----------------
        BUFFER.store(states_py[idx], int(act), r_t, float(v_t), float(logp))

    # ------------------------------------------------------------------ #
    # 3) when enough transitions → one gradient step
    # ------------------------------------------------------------------ #
    if BUFFER.size() >= ROLL_OUT:
        update_a2c(actor, critic, optimizer)
        BUFFER.clear()

################################################################################
#  UPDATE FUNCTION
################################################################################
def compute_gae(rewards, values, gamma, lam):
    adv, gae = np.zeros_like(rewards), 0.0
    next_v = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*next_v - values[t]
        gae   = delta + gamma*lam*gae
        adv[t]= gae
        next_v= values[t]
    returns = adv + values
    return adv, returns

###############################################################################
#  TRAIN-STEP  (runs under @tf.function for speed)
###############################################################################
@tf.function
def _train_step(actor, critic, opt,
                s_batch, a_batch,
                adv_batch, ret_batch,
                entropy_coeff_f64):

    # cast the scalar → float32 so all operands match
    entropy_coeff = tf.cast(entropy_coeff_f64, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        logits = actor(s_batch)                           # (B,2)
        logp   = tf.nn.log_softmax(logits)                # (B,2)
        act_logp = tf.gather(logp, tf.expand_dims(a_batch, 1), batch_dims=1)
        pg_loss  = -tf.reduce_mean(act_logp * adv_batch)

        values   = tf.squeeze(critic(s_batch), axis=1)    # (B,)
        v_loss   = tf.reduce_mean(tf.square(ret_batch - values))

        entropy  = -tf.reduce_mean(
            tf.reduce_sum(tf.exp(logp) * logp, axis=1)
        )

        loss = pg_loss + 0.5 * v_loss - entropy_coeff * entropy

    # apply gradients to actor & critic separately
    actor_grads  = tape.gradient(loss, actor.trainable_variables)
    critic_grads = tape.gradient(loss, critic.trainable_variables)

    # NEW unified apply_gradients call
    grads_and_vars = (
        list(zip(actor_grads,  actor.trainable_variables)) +
        list(zip(critic_grads, critic.trainable_variables))
    )
    opt.apply_gradients(grads_and_vars)


###############################################################################
#  UPDATE A2C  (called when the rollout buffer reaches ROLL_OUT samples)
###############################################################################
def update_a2c(actor, critic, optimizer):
    global GLOBAL_STEP

    # ---------- pull & convert the rollout ----------
    s   = tf.convert_to_tensor(np.array(BUFFER.s), dtype=tf.float32)
    a   = tf.convert_to_tensor(np.array(BUFFER.a), dtype=tf.int32)
    r   = np.array(BUFFER.r, dtype=np.float32)
    v   = np.array(BUFFER.v, dtype=np.float32)

    # ---------- GAE + returns ----------
    adv, ret = compute_gae(r, v, GAMMA, LAMBDA)
    adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

    # cast to tensors once
    adv  = tf.convert_to_tensor(adv,  dtype=tf.float32)
    ret  = tf.convert_to_tensor(ret,  dtype=tf.float32)

    # ---------- entropy-annealing ----------
    step          = tf.cast(GLOBAL_STEP, tf.float32)
    entropy_coeff = ENTROPY_MIN + (ENTROPY_MAX - ENTROPY_MIN) * \
                    tf.exp(-step / ENTROPY_DECAY_STEPS)
    GLOBAL_STEP.assign_add(1)

    # ---------- single optimisation step ----------
    _train_step(actor, critic, optimizer,
                s, a, adv, ret, entropy_coeff)

def warmup_models(actor, critic, optimizer, input_dim):
    dummy_state = tf.zeros((1, input_dim))
    actor(dummy_state)
    critic(dummy_state)
    with tf.GradientTape() as tape:
        logits = actor(dummy_state)
        value = critic(dummy_state)
        dummy_loss = tf.reduce_sum(logits) + tf.reduce_sum(value)
    grads = tape.gradient(dummy_loss, actor.trainable_variables + critic.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables + critic.trainable_variables))
