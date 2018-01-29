import functools
import time


def retry(timeoutSec, initialRetryDelaySec, maxRetryDelaySec,
          retryExceptions=(Exception,),
          retryFilter=lambda e, args, kwargs: True):
    """ Returns a closure suitable for use as function/method decorator for
    retrying a function being decorated.

    :param timeoutSec: How many seconds from time of initial call to stop
        retrying (floating point); 0 = no retries
    :param initialRetryDelaySec: Number of seconds to wait for first retry.
        Subsequent retries will occur at geometrically doubling intervals up to
        a maximum interval of maxRetryDelaySec (floating point)
    :param maxRetryDelaySec: Maximum amount of seconds to wait between retries
        (floating point)
    :param retryExceptions: A tuple (must be a tuple) of exception classes that,
        including their subclasses, should trigger retries; Default: any
        Exception-based exception will trigger retries
    :param retryFilter: Optional filter function used to further filter the
        exceptions in the retryExceptions tuple; called if the current
        exception meets the retryExceptions criteria: takes the current
        exception instance, args, and kwargs that were passed to the decorated
        function, and returns True to retry, False to allow the exception to be
        re-raised without retrying. Default: permits any exception that matches
        retryExceptions to be retried.

    Example:
    ::

        _retry = retry(timeoutSec=300, initialRetryDelaySec=0.2,
        maxRetryDelaySec=10, retryExceptions=[socket.error])
        @_retry
        def myFunctionFoo():
            ...
            raise RuntimeError("something bad happened")
            ...
    """
    assert initialRetryDelaySec > 0, str(initialRetryDelaySec)
    assert timeoutSec >= 0, str(timeoutSec)
    assert maxRetryDelaySec >= initialRetryDelaySec, \
        "%r < %r" % (maxRetryDelaySec, initialRetryDelaySec)

    assert isinstance(retryExceptions, tuple), (
        "retryExceptions must be tuple, but got %r") % (type(retryExceptions),)

    def retryDecorator(func):

        @functools.wraps(func)
        def retryWrap(*args, **kwargs):
            numAttempts = 0
            delaySec = initialRetryDelaySec
            startTime = time.time()

            # Make sure it gets called at least once
            while True:
                numAttempts += 1
                try:
                    result = func(*args, **kwargs)
                except retryExceptions as e:
                    if not retryFilter(e, args, kwargs):
                        raise e

                    now = time.time()
                    # Compensate for negative time adjustment so we don't get
                    # stuck waiting way too long (python doesn't provide
                    # monotonic time yet)
                    if now < startTime:
                        startTime = now
                    if (now - startTime) >= timeoutSec:
                        raise e

                    time.sleep(delaySec)

                    delaySec = min(delaySec * 2, maxRetryDelaySec)
                else:

                    return result

        return retryWrap

    return retryDecorator
