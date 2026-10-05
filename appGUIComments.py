"""
self._running = False

1:58 PM
When root.destroy() is called, tkinter deletes the window and all its internal commands. But _animate schedules itself with root.after(FPS_MS, self._animate) — so there's always a queued callback sitting in tkinter's event queue waiting to fire.

The problem is the timing: root.destroy() runs, deletes everything, but that queued after callback was already registered and fires a split second later — except now the window it's trying to talk to no longer exists. Tkinter throws the invalid command name error because it can't find the callback anymore.

self._running = False is just a flag that tells _animate — before it does anything or schedules itself again — "stop, we're done." So instead of crashing into a destroyed window, it just quietly returns.




"""