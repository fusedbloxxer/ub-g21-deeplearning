from functools import wraps


def hide_self(procedure):
    """Hide `self` instance inside trial object."""
    @wraps(procedure)
    def hidden(self, trial):
        trial.inner = self
        return procedure(trial)
    return hidden


def show_self(procedure):
    """Retrieve `self` from trial object and forward it to the wrapped __call__."""
    @wraps(procedure)
    def hidden(trial):
        return procedure(trial.inner, trial)
    return hidden


def forward_self(callback):
    """Wrap an Optuna callback and hide & retrieve the `self` instance."""
    def forward_inner(procedure):
        @wraps(procedure)
        def hidden(*args, **kwargs):
            return hide_self(callback(show_self(procedure)))(*args, **kwargs)
        return hidden
    return forward_inner
