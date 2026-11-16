"""Loops with a static trip count, so jax.grad / vjp can go through them.

jax.lax.while_loop, and fori_loop with a traced bound, support forward-mode
AD only. These replacements run a lax.scan of static length max_iter whose
body is lax.cond(active, body, identity): skipped iterations are identities
with identity derivatives. On CPU outside vmap, cond runs only the taken
branch, so skipped iterations are cheap; under vmap both branches run.

checkpoint=True wraps the body in jax.checkpoint: the backward pass stores
one carry per iteration and recomputes the body's intermediates.

The arithmetic inside the body is the loop's own, but XLA may fuse it
differently inside cond, so results can differ from the unbounded loop by
round-off (see tests/test_reverse_mode.py).
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp


def bounded_while(cond_fn, body_fn, init, max_iter, checkpoint=True):
    """while_loop(cond_fn, body_fn, init) with at most max_iter iterations."""
    body = jax.checkpoint(body_fn) if checkpoint else body_fn

    def step(state, _):
        return jax.lax.cond(cond_fn(state), body, lambda s: s, state), None

    out, _ = jax.lax.scan(step, init, None, length=max_iter)
    return out


def bounded_fori(lower, upper, body_fn, init, max_iter, checkpoint=False):
    """fori_loop(lower, upper, body_fn, init) for upper - lower <= max_iter.

    lower and upper may be traced; iterations i >= lower + max_iter are
    dropped.
    """
    def body(i, state):
        return body_fn(i, state)
    body = jax.checkpoint(body) if checkpoint else body

    def step(state, k):
        i = lower + k
        return jax.lax.cond(i < upper, lambda s: body(i, s), lambda s: s,
                            state), None

    out, _ = jax.lax.scan(step, init, jnp.arange(max_iter))
    return out


def loop_while(cond_fn, body_fn, init, bounded=False, max_iter=None,
               checkpoint=True):
    """lax.while_loop, or bounded_while(max_iter) when bounded (static)."""
    if bounded:
        return bounded_while(cond_fn, body_fn, init, max_iter, checkpoint)
    return jax.lax.while_loop(cond_fn, body_fn, init)


def loop_fori(lower, upper, body_fn, init, bounded=False, max_iter=None,
              checkpoint=False):
    """lax.fori_loop, or bounded_fori(max_iter) when bounded (static)."""
    if bounded:
        return bounded_fori(lower, upper, body_fn, init, max_iter, checkpoint)
    return jax.lax.fori_loop(lower, upper, body_fn, init)
