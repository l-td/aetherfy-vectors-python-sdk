"""Collection storage modes.

A collection's storage mode is a preset for how its vectors are stored and
searched. The service compiles it to the engine's configuration; the caller
never sends engine tuning directly (see ``aetherfy_vectors.qdrant_compat``,
which refuses ``quantization_config`` and the other storage settings).

=========  ===================================================================
Mode       What it does
=========  ===================================================================
exact      The default: full-precision vectors in memory, search reads them.
balanced   An 8-bit copy of each vector is searched, the originals sit on disk
           and rescore the best candidates. Uses far less memory than exact.
fast       The same 8-bit copy, with the originals kept in memory too. Uses
           MORE memory than exact, not less.
compact    A 2-bit copy, originals on disk. The smallest footprint and the
           lossiest mode; needs vectors of at least 512 dimensions.
=========  ===================================================================

The same names are accepted by the JS SDK (``storageMode``).
"""

from enum import Enum
from typing import Optional, Union


class StorageMode(str, Enum):
    """A collection's storage mode. Compares equal to its string value."""

    EXACT = "exact"
    BALANCED = "balanced"
    FAST = "fast"
    COMPACT = "compact"


#: What a collection created without ``storage_mode`` gets.
DEFAULT_STORAGE_MODE = StorageMode.EXACT


def coerce_storage_mode(value: Union[StorageMode, str]) -> StorageMode:
    """Return ``value`` as a StorageMode.

    Raises:
        ValueError: if it is not one of the four modes. A typo fails here,
            before a round trip, and names the accepted values.
    """
    if isinstance(value, StorageMode):
        return value
    if isinstance(value, str):
        try:
            return StorageMode(value)
        except ValueError:
            pass
    raise ValueError(
        f"Invalid storage_mode: {value!r}. "
        f"Must be one of: {', '.join(m.value for m in StorageMode)}"
    )


def parse_storage_mode(value: object) -> Optional[str]:
    """Read a mode off a server response.

    A known mode becomes the enum. A name this SDK has not heard of (a newer
    server) is returned as the plain string rather than raising, so listing
    collections keeps working across a server upgrade. ``None`` (an older
    server that does not report modes) stays ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return StorageMode(value)
        except ValueError:
            return value
    return None
