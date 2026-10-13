"""A6000-only entry point for the NVIDIA 535 Vulkan shutdown crash."""
import atexit
import os
import sys


def main():
    completed = False

    def finish():
        if completed:
            sys.stdout.flush()
            sys.stderr.flush()
            # shortcut: skip crashing native destructors; remove after a validated driver fix.
            os._exit(0)

    # Register first so simulation/Python cleanup runs before this final handler.
    atexit.register(finish)
    from robofollow.evaluate import main as evaluate
    evaluate()
    completed = True


if __name__ == '__main__':
    main()
