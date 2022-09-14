from threading import Lock

#Detector metaclass that is used as a singleton
class DiagramInterpreterMeta(type):
    _instances = {}

    _lock: Lock = Lock()

    def __call__(self, *args, **kwds):
        with self._lock:
            if self not in self._instances:
                instance = super().__call__(*args, *kwds)
                self._instances[self] = instance
        return self._instances[self]
