import math

import tensorflow as tf


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warmup_steps=0, name=None):
        """
        Applies a cosine decay schedule with a linear warmup phase.

        Parameters:
        - initial_learning_rate: A scalar `float32` or `float64` Tensor or a
          Python number. The initial learning rate.
        - decay_steps: A scalar `int32` or `int64` Tensor or a Python number.
          Number of steps to decay over.
        - alpha: A scalar `float32` or `float64` Tensor or a Python number.
          Minimum learning rate value as a fraction of initial_learning_rate.
        - warmup_steps: A scalar `int32` or `int64` Tensor or a Python number.
          Number of steps to linearly warmup learning rate from 0 to initial_learning_rate.
        - name: Optional name prefix for the operations created when applying
          gradients. Defaults to 'CosineDecayWithWarmup'.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecayWithWarmup"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            global_step_recomp = tf.minimum(global_step_recomp, decay_steps)

            # Linear warmup
            warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
            warmup_phase = tf.cast(step < warmup_steps, dtype)

            # Cosine decay
            cosine_decay = 0.5 * (1 + tf.cos(math.pi * global_step_recomp / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            cosine_lr = initial_learning_rate * decayed

            # Choose warmup phase or cosine decay phase
            learning_rate = (warmup_lr * warmup_phase) + (cosine_lr * (1 - warmup_phase))

            return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }


def main():
    scheduler = CosineDecayWithWarmup(1.0, 100, warmup_steps=10)

    vals = []

    for step in range(100):
        lr = scheduler(step).numpy()

        vals.append([step, lr])

    import pandas as pd

    data = pd.DataFrame(vals, columns=['step', 'lr'])



if __name__ == '__main__':
    main()
