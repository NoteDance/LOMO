# LOMO

**Overview**:

The `LOMO` (LOw-Memory Optimization) optimizer is designed to reduce the memory footprint during training, particularly in large-scale distributed settings like ZeRO stage 3. It achieves this by fusing the gradient computation and parameter update steps for each parameter individually, avoiding the need to store all gradients simultaneously. It supports gradient clipping by norm or value and incorporates optional dynamic loss scaling for mixed-precision training (`tf.float16`).

**Parameters**:

-   **`model`** *(tf.keras.Model)*: The Keras model whose parameters will be optimized.
-   **`lr`** *(float, default=1e-3)*: The learning rate or step size for parameter updates.
-   **`clip_grad_norm`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped globally based on the total norm of all gradients. Requires calling `optimizer.grad_norm()` before `optimizer.fused_backward()`.
-   **`clip_grad_value`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped element-wise to stay within the range `[-clip_grad_value, +clip_grad_value]`.
-   **`zero3_enabled`** *(bool, default=True)*: If `True`, enables ZeRO stage 3 style optimization where gradients are reduced across replicas and only the relevant partition of the parameter is updated locally. If `False`, performs standard updates on the full parameters.
-   **`name`** *(str, default="lomo")*: The name for the optimizer instance.

*(Note: For `tf.float16` parameters, dynamic loss scaling is automatically enabled to prevent underflow.)*

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lomo import LOMO # Assuming LOMO is in your_module

# --- Model Definition ---
# inputs = tf.keras.Input(shape=(...))
# outputs = YourModelLayers(inputs)
# model = tf.keras.Model(inputs, outputs)
# ------------------------

# Instantiate optimizer
optimizer = LOMO(model, lr=1e-3, clip_grad_norm=1.0)

# --- Training Step ---
# @tf.function # Decorate for performance
def train_step(inputs, labels):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        # Ensure loss computation uses float32 for stability
        loss = tf.keras.losses.your_loss_function(labels, tf.cast(predictions, tf.float32))

    # If clip_grad_norm is used, calculate norm first
    if optimizer.clip_grad_norm is not None and optimizer.clip_grad_norm > 0.0:
         # Pass loss before potential scaling if LOMO handles it internally
         # Note: The provided LOMO code seems to scale loss *inside* grad_norm/fused_backward
         optimizer.grad_norm(tape, loss, model.trainable_variables)
         # fused_backward will use the calculated clip_coef

    # Perform fused backward pass and update
    # Pass the original, potentially unscaled loss if LOMO handles scaling
    optimizer.fused_backward(tape, loss, model.trainable_variables, lr=optimizer.lr) # Pass current lr

    return loss
# ---------------------

# --- Training Loop ---
# for epoch in range(num_epochs):
#     for step, (x_batch, y_batch) in enumerate(train_dataset):
#         loss_value = train_step(x_batch, y_batch)
#         if step % log_interval == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.numpy()}")
# ---------------------
```

*(Note: LOMO requires a custom training loop because it uses `fused_backward` and potentially `grad_norm` instead of the standard Keras `optimizer.apply_gradients` used within `model.compile`/`model.fit`.)*

# AdaLOMO

**Overview**:

The `AdaLOMO` optimizer combines the low-memory optimization strategy of LOMO with adaptive learning rate methods. It approximates the second moment of gradients using row and column averages (similar to Adafactor) to adapt the learning rate for each parameter, aiming for improved convergence and stability while maintaining low memory usage. It includes features like weight decay, adaptive gradient clipping based on update norms, and learning rate scaling based on parameter norms (similar to LAMB).

**Parameters**:

-   **`model`** *(tf.keras.Model)*: The Keras model whose parameters will be optimized.
-   **`lr`** *(float, default=1e-3)*: The base learning rate. The actual step size is adapted based on parameter norms and second moment estimates.
-   **`weight_decay`** *(float, default=0.0)*: Coefficient for L2 weight decay (applied additively to the gradient before the update step).
-   **`loss_scale`** *(float, default=1024.0)*: Static loss scaling factor used to prevent gradient underflow, especially during mixed-precision training. Gradients are unscaled internally before updates.
-   **`clip_threshold`** *(float, default=1.0)*: Threshold for adaptive gradient clipping. The normalized update is clipped based on this value.
-   **`decay_rate`** *(float, default=-0.8)*: Exponent used to compute the decay factor (`beta2_t`) for the running averages of squared gradients. `beta2_t = 1.0 - steps ** decay_rate`.
-   **`clip_grad_norm`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped globally based on the total norm of all gradients *before* adaptive calculations. Requires calling `optimizer.grad_norm()` before `optimizer.fused_backward()`.
-   **`clip_grad_value`** *(float, optional, default=None)*: If set to a positive value, gradients will be clipped element-wise *before* adaptive calculations.
-   **`eps1`** *(float, default=1e-30)*: A small epsilon added to the squared gradients before computing row/column means, ensuring numerical stability.
-   **`eps2`** *(float, default=1e-3)*: A small epsilon used when scaling the learning rate by the parameter norm (`lr_scaled = lr * max(eps2, p_rms)`), preventing division by zero or overly large learning rates for small parameters.
-   **`zero3_enabled`** *(bool, default=True)*: If `True`, enables ZeRO stage 3 style optimization where gradients are reduced, second moments are potentially calculated on full gradients, and only the relevant partition of the parameter is updated locally using partitioned updates. If `False`, performs standard updates on the full parameters.
-   **`name`** *(str, default="adalomo")*: The name for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from optimizers.lomo import AdaLOMO # Assuming AdaLOMO is in your_module

# --- Model Definition ---
# inputs = tf.keras.Input(shape=(...))
# outputs = YourModelLayers(inputs)
# model = tf.keras.Model(inputs, outputs)
# ------------------------

# Instantiate optimizer
optimizer = AdaLOMO(
    model,
    lr=1e-3,
    weight_decay=0.01,
    clip_threshold=1.0,
    clip_grad_norm=1.0 # Example: enabling global grad norm clipping
)

# --- Training Step ---
# @tf.function # Decorate for performance
def train_step(inputs, labels):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        # Ensure loss computation uses float32 for stability
        loss = tf.keras.losses.your_loss_function(labels, tf.cast(predictions, tf.float32))

    # If clip_grad_norm is used, calculate norm first
    if optimizer.clip_grad_norm is not None and optimizer.clip_grad_norm > 0.0:
        # Pass loss before potential scaling if AdaLOMO handles it internally
        # Note: The provided AdaLOMO code scales loss *inside* grad_norm/fused_backward
        optimizer.grad_norm(tape, loss, model.trainable_variables)
        # fused_backward will use the calculated clip_coef

    # Perform fused backward pass and update
    # Pass the original, potentially unscaled loss if AdaLOMO handles scaling
    optimizer.fused_backward(tape, loss, model.trainable_variables, lr=optimizer.lr) # Pass base lr

    return loss
# ---------------------

# --- Training Loop ---
# optimizer.num_steps = 0 # Initialize step counter
# for epoch in range(num_epochs):
#     for step, (x_batch, y_batch) in enumerate(train_dataset):
#         loss_value = train_step(x_batch, y_batch)
#         if step % log_interval == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.numpy()}")
# ---------------------
```
