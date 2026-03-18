# debug_entry.py — set up diagnostics before your app starts
import faulthandler
import logging
import signal
import sys
import threading
import traceback


# Make sure we see thread exceptions (Python 3.8+)
def _thread_excepthook(args: threading.ExceptHookArgs):
    logging.error(
        "THREAD CRASH: %s",
        args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _thread_excepthook

# Always be able to dump all stacks: kill -USR1 <pid>
faulthandler.register(signal.SIGUSR1)

# Instrument fsspec LocalFileOpener.closed so we SEE the callsite when f is None
try:
    import fsspec.implementations.local as _fsl

    _orig_closed = getattr(_fsl.LocalFileOpener, "closed", None)

    def _closed_with_trace(self):
        f = getattr(self, "f", None)
        if f is None:
            logging.error(
                "fsspec LocalFileOpener.closed called with f=None\n%s",
                "".join(traceback.format_stack(limit=50)),
            )
            return True  # also make it defensive so it doesn't crash
        return f.closed

    if isinstance(_orig_closed, property):
        _fsl.LocalFileOpener.closed = property(_closed_with_trace)  # type: ignore
except Exception as e:
    logging.getLogger(__name__).warning("fsspec patch failed: %r", e)

# Optional: loud asyncio loop errors
try:
    import asyncio

    def _aio_handler(loop, context):
        logging.error(
            "ASYNCIO ERROR: %s",
            context.get("message"),
            exc_info=context.get("exception"),
        )

    asyncio.get_event_loop().set_exception_handler(_aio_handler)
except Exception:
    pass

# Hand over to main (which dispatches to library)
sys.argv[0] = "main.py"
from main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
