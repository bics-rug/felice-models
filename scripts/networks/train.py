import argparse
import os
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import pandas as pd
from jaxtyping import Array, Float
from optax import OptState
from tqdm import trange

from felice.datasets.reasoning import ReasoningDataset
from felice.networks import Implicit, Mamba, SequenceClassifier
from felice.networks.implicit.boomerang import ImplicitBoomerang


def compute_loss(
    model: eqx.Module, inputs: Array, targets: Array, masks: Array
) -> Float[Array, ""]:
    def forward_single(inp, tgt, msk):
        logits = model(inp)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
        return (loss * msk).sum() / (msk.sum() + 1e-8)

    losses = jax.vmap(forward_single)(inputs, targets, masks)
    return losses.mean()


v_and_grad = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def compute_accuracy(
    model: eqx.Module, inputs: Array, targets: Array, masks: Array
) -> Float[Array, ""]:
    def forward_single(inp, tgt, msk):
        logits = model(inp)
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == tgt) * msk
        return correct.sum(), msk.sum()

    correct, total = jax.vmap(forward_single)(inputs, targets, masks)
    return correct.sum() / (total.sum() + 1e-8)


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt_state: OptState,
    optimizer: Any,
    inputs: Array,
    targets: Array,
    masks: Array,
) -> Tuple[eqx.Module, OptState, Array]:
    loss, grads = v_and_grad(model, inputs, targets, masks)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_and_compare(
    model_type: Any,
    logdir: str,
    task_type: str = "simple",
    n_epochs: int = 1000,
    batch_size: int = 64,
    d_model: int = 64,
    d_state: int = 16,
    d_inner: int = 32,
    dt: float = 1.0,
    max_iters: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
    # with_thr: bool = True,
) -> Tuple[eqx.Module, eqx.Module, Array, Array, pd.DataFrame]:
    r"""Train Mamba and implicit model on the reasoning synthetic dataset.

    Args:
        model_type: The type of the implicit model to train (Boomerang, Mamba Implicit).
        logdir: Directory and filenmae of the log.
        task_type: Type of task to solve from the reasoning synthetic dataset (simple, accumulation).
        n_epochs: Number of epochs to train.
        batch_size: Training batch size.
        d_model: Model dimensions including output.
        d_state: Model state dimension.
        d_inner: Model latent dimension.
        max_iters: Maximum number of iterations in the implicit model.
        lr: Learning rate.
        seed: Random seed.
        with_thr: For the Boomerang model, if using threshold for dual fixpoints.

    Returns:
        The trained models (mamba and implicit) with the respective final accuracy and
        a pandas dataframe with the loss and accuracy per epoch.
    """
    key = jrandom.key(seed)
    keys = jrandom.split(key, 4)

    dataset = ReasoningDataset()

    standard_model = SequenceClassifier(
        vocab_size=dataset.VOCAB_SIZE,
        d_model=d_model,
        d_state=d_state,
        d_inner=d_inner,
        model_class=Mamba,
        key=keys[0],
    )

    implicit_model = SequenceClassifier(
        vocab_size=dataset.VOCAB_SIZE,
        d_model=d_model,
        d_state=d_state,
        d_inner=d_inner,
        model_class=model_type,
        max_iters=max_iters,
        dt=dt,
        # with_thr=with_thr,
        key=keys[1],
    )

    # implicit_model = ImplicitBoomerang(
    #     vocab_size=dataset.VOCAB_SIZE,
    #     d_model=d_model,
    #     d_state=d_state,
    #     d_inner=d_inner,
    #     max_iters=max_iters,
    #     dt=dt,
    #     # with_thr=with_thr,
    #     key=keys[1],
    # )
    # Count parameters
    def count_params(model):
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )

    print(f"Mamba SSM params: {count_params(standard_model):,}")
    print(f"Implicit SSM params: {count_params(implicit_model):,}")

    # Optimizers
    optimizer = optax.adam(lr)
    standard_opt_state = optimizer.init(eqx.filter(standard_model, eqx.is_array))
    implicit_opt_state = optimizer.init(eqx.filter(implicit_model, eqx.is_array))

    # Training loop
    print(f"\nTraining on task: {task_type} with {max_iters} steps")
    print("=" * 60)

    train_key = keys[2]

    df = pd.DataFrame({"Epoch": [], "Loss": [], "Acc": [], "Model": []})
    pbar = trange(n_epochs)
    for epoch in pbar:
        train_key, batch_key = jrandom.split(train_key)
        inputs, targets, masks = dataset.generate_batch(
            batch_key, batch_size, task_type
        )

        # Train standard model
        standard_model, standard_opt_state, standard_loss = train_step(
            standard_model, standard_opt_state, optimizer, inputs, targets, masks
        )

        # Train implicit model
        implicit_model, implicit_opt_state, implicit_loss = train_step(
            implicit_model, implicit_opt_state, optimizer, inputs, targets, masks
        )

        if (epoch + 1) % 10 == 0:
            # Evaluate on fresh batch
            eval_key = jrandom.fold_in(keys[3], epoch)
            eval_inputs, eval_targets, eval_masks = dataset.generate_batch(
                eval_key, batch_size, task_type
            )

            standard_acc = compute_accuracy(
                standard_model, eval_inputs, eval_targets, eval_masks
            )
            implicit_acc = compute_accuracy(
                implicit_model, eval_inputs, eval_targets, eval_masks
            )

            new_df = pd.DataFrame(
                {
                    "Epoch": [epoch, epoch],
                    "Loss": [standard_loss.item(), implicit_loss.item()],
                    "Acc": [standard_acc.item(), implicit_acc.item()],
                    "Model": ["Mamba", "Implicit"],
                }
            )
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_pickle(logdir)
            pbar.write(
                f"Epoch {epoch + 1:4d} | "
                f"Standard: loss={standard_loss:.4f}, acc={standard_acc:.4f} | "
                f"Implicit: loss={implicit_loss:.4f}, acc={implicit_acc:.4f}"
            )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (1000 samples)")
    print("=" * 60)

    eval_inputs, eval_targets, eval_masks = dataset.generate_batch(
        keys[4], 1000, task_type
    )

    standard_acc = compute_accuracy(
        standard_model, eval_inputs, eval_targets, eval_masks
    )
    implicit_acc = compute_accuracy(
        implicit_model, eval_inputs, eval_targets, eval_masks
    )

    print(f"Mamba SSM accuracy: {standard_acc:.4f}")
    print(f"Implicit SSM accuracy: {implicit_acc:.4f}")
    print(f"Improvement: {(implicit_acc - standard_acc) * 100:.2f}%")

    return standard_model, implicit_model, standard_acc, implicit_acc, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", type=int, choices=[1, 2], default=1, help="Task to perform"
    )
    parser.add_argument(
        "-m",
        type=str,
        choices=["boomerang", "implicit"],
        default="implicit",
        help="Neuron model to use",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument("-i", type=int, default=8, help="Maximum number of iterations")
    parser.add_argument("-b", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--thr", action="store_true", help="Using threshold on the boomerang neuron"
    )
    args = parser.parse_args()

    if args.m == "boomerang":
        model_type = ImplicitBoomerang
    elif args.m == "implicit":
        model_type = Implicit
    else:
        raise NotImplementedError(f"{args.t} model type not implemented")

    logdir = os.path.join("tmp", f"task{args.t}-{args.i}-{args.m}-{args.thr}")
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    print(f"Saving at {logdir}")
    _, implicit_model, std_acc1, imp_acc1, df = train_and_compare(
        model_type,
        logdir,
        task_type="simple" if args.t == 1 else "accumulation",
        n_epochs=1000,
        batch_size=64,
        d_model=ReasoningDataset.NUM_OUTPUT,
        d_state=16,
        d_inner=128,
        dt=args.dt,
        max_iters=args.i,
        # with_thr=args.thr,
    )
    eqx.tree_serialise_leaves(f"{logdir}.eqx", implicit_model)
    df.to_pickle(logdir)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Task':<25} {'Mamba SSM':<15} {'Implicit SSM':<15} {'Delta':<10}")
    print("-" * 70)
    print(
        f"{'Simple Comparison':<25} {std_acc1:<15.4f} {imp_acc1:<15.4f} {(imp_acc1 - std_acc1) * 100:>+.2f}%"
    )
