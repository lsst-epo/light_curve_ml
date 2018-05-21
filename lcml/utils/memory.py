import logging
import os

from psutil import Process


logger = logging.getLogger(__name__)


_BYTES_IN_GB = 10e9


def reportProcessMemoryUsage():
    """
    rss: aka "Resident Set Size", non-swapped physical memory process has used.
    The portion of memory occupied by a process that is held in main memory
    (RAM). The rest of the occupied memory exists in the swap space or file
    system, either because some parts of the occupied memory were paged out, or
    because some parts of the executable were never loaded

    vms: aka "Virtual Memory Size", total amount of virtual memory used by the
    process.
    """
    p = Process(os.getpid())
    mi = p.memory_info()
    _rssGb = mi.rss / _BYTES_IN_GB
    _vmsGb = mi.vms / _BYTES_IN_GB
    logger.info("ram %.2f GB vm: %.2f GB", _rssGb, _vmsGb)
