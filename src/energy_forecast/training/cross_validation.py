"""Time-series aware cross-validation splitters."""

from __future__ import annotations

from typing import Generator, Literal

import numpy as np
from numpy.typing import NDArray

from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

_Strategy = Literal["expanding_window", "sliding_window", "blocked"]


class TimeSeriesCV:
    """Cross-validation splitter that respects temporal ordering.

    Unlike scikit-learn's ``TimeSeriesSplit``, this class offers three
    strategies and an explicit *gap* parameter to prevent data leakage
    caused by autocorrelation.

    Parameters
    ----------
    n_splits:
        Number of train/test folds to generate.
    strategy:
        One of ``"expanding_window"``, ``"sliding_window"``, or
        ``"blocked"``.
    gap:
        Number of samples to skip between the training and test windows.
        Defaults to ``24`` (one day of hourly data) so that highly
        correlated neighbouring observations do not leak information.
    """

    VALID_STRATEGIES: tuple[str, ...] = ("expanding_window", "sliding_window", "blocked")

    def __init__(
        self,
        n_splits: int = 5,
        strategy: _Strategy = "expanding_window",
        gap: int = 24,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {self.VALID_STRATEGIES}"
            )
        if gap < 0:
            raise ValueError("gap must be >= 0")

        self.n_splits = n_splits
        self.strategy = strategy
        self.gap = gap

    def get_n_splits(self) -> int:
        """Return the number of folds."""
        return self.n_splits

    def split(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | None = None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate train/test index pairs.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, ...)``.  Only the first
            axis is used for determining split boundaries.
        y:
            Ignored.  Present for API compatibility.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            ``(train_indices, test_indices)`` for each fold.
        """
        n_samples = len(X)

        if self.strategy == "expanding_window":
            yield from self._expanding_window(n_samples)
        elif self.strategy == "sliding_window":
            yield from self._sliding_window(n_samples)
        elif self.strategy == "blocked":
            yield from self._blocked(n_samples)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _expanding_window(
        self, n_samples: int
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Training window starts at index 0 and grows; test is fixed size.

        The available data after accounting for the gap is divided into
        ``n_splits + 1`` segments.  For each fold *k* (0-indexed) the
        training set covers segments ``0 .. k`` and the test set is
        segment ``k + 1``.
        """
        # Reserve a minimum training size of one segment
        total_usable = n_samples
        # Each fold needs: test_size + gap, and initial train needs at least test_size
        test_size = max(1, total_usable // (self.n_splits + 1))

        for k in range(self.n_splits):
            train_end = test_size * (k + 1)
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                # Clip the last fold to available data
                test_end = n_samples
            if test_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            logger.debug(
                "cv_fold",
                strategy=self.strategy,
                fold=k,
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
            yield train_idx, test_idx

    def _sliding_window(
        self, n_samples: int
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Fixed-size training window slides forward through the data.

        The training window size is determined as a fraction of the total
        data, and both training and test windows advance by the same step
        each fold.
        """
        test_size = max(1, n_samples // (self.n_splits + 1))
        train_size = test_size * 2  # train is twice the test window

        # Ensure train_size fits
        if train_size + self.gap + test_size > n_samples:
            train_size = max(1, n_samples - self.gap - test_size * self.n_splits)

        step = max(1, (n_samples - train_size - self.gap - test_size) // max(1, self.n_splits - 1))

        for k in range(self.n_splits):
            train_start = k * step
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                test_end = n_samples
            if test_start >= n_samples or train_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            logger.debug(
                "cv_fold",
                strategy=self.strategy,
                fold=k,
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
            yield train_idx, test_idx

    def _blocked(
        self, n_samples: int
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Equal-size non-overlapping blocks; each block is used as test once.

        Data is divided into ``n_splits + 1`` contiguous blocks.  For each
        fold the test block is one of blocks ``1 .. n_splits`` and all
        preceding blocks (minus the gap) form the training set.
        """
        block_size = max(1, n_samples // (self.n_splits + 1))

        for k in range(self.n_splits):
            test_start = block_size * (k + 1)
            test_end = min(test_start + block_size, n_samples)

            train_end = test_start - self.gap
            if train_end <= 0 or test_start >= n_samples:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            logger.debug(
                "cv_fold",
                strategy=self.strategy,
                fold=k,
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"TimeSeriesCV(n_splits={self.n_splits}, "
            f"strategy='{self.strategy}', gap={self.gap})"
        )
