import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import requests

# ── Screen config (auto-detected at runtime) ──────────────────────────────────
WIN_W, WIN_H = 1920, 1080
SERVER_URL = os.environ.get('AI_DETECTOR_SERVER_URL', 'http://localhost:8000')
CONTACT_EMAIL = 'cristellodominic@gmail.com'
TURQUOISE = '#40E0D0'
FIELD_ERROR = '#EF4444'


# ── Gradient background (identical to startup GUI) ────────────────────────────
def make_gradient(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([4, 4, 14], dtype=np.float32)
    img  = np.tile(base, (h, w, 1))

    dx = (xx - w * 0.70) / (w * 0.45)
    dy = (yy - h * 0.38) / (h * 0.55)
    blue = np.clip(1.0 - np.sqrt(dx**2 + dy**2), 0, 1) ** 2.2
    img[:,:,0] += blue * 12;  img[:,:,1] += blue * 35;  img[:,:,2] += blue * 130

    dx2 = (xx - w * 0.25) / (w * 0.38)
    dy2 = (yy - h * 0.78) / (h * 0.45)
    purple = np.clip(1.0 - np.sqrt(dx2**2 + dy2**2), 0, 1) ** 1.8
    img[:,:,0] += purple * 90; img[:,:,1] += purple * 8; img[:,:,2] += purple * 110

    dx3 = (xx - w * 0.10) / (w * 0.30)
    dy3 = (yy - h * 0.15) / (h * 0.30)
    teal = np.clip(1.0 - np.sqrt(dx3**2 + dy3**2), 0, 1) ** 2.5
    img[:,:,0] += teal * 0;  img[:,:,1] += teal * 30;  img[:,:,2] += teal * 60

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ── Plan data ─────────────────────────────────────────────────────────────────
PLANS = [
    {
        'name':         'Basic',
        'price':        'Free',
        'cents':        '',
        'tagline':      'For the casual listener',
        'btn_text':     'Choose Basic',
        'btn_color':    '#354052',
        'btn_hover':    '#46556B',
        'btn_text_col': '#FFFFFF',
        'card_bg':      '#0C0C1E',
        'border':       '#1E2A3A',
        'glow':         False,
        'free':         True,
        'features': [
            '3 songs per month',
            'Gauge meter result',
            'File-generated waveform',
            'Min 60s  /  Max 600s duration',
            'Single agent processing',
        ],
    },
    {
        'name':         'Premium',
        'price':        'Free',
        'cents':        '',
        'tagline':      'For professional music enthusiasts',
        'btn_text':     'Choose Premium',
        'btn_color':    '#38BDF8',
        'btn_hover':    '#0EA5E9',
        'btn_text_col': '#0C1A2E',
        'card_bg':      '#0A0A22',
        'border':       '#38BDF8',
        'glow':         True,
        'free':         True,
        'features': [
            'Unlimited songs per month',
            'Gauge meter + information dropdown',
            'Saved detection history',
            'File-generated spectrogram + waveform',
            'Min 60s  /  Max 1200s duration',
            'Single agent processing',
        ],
    },
    {
        'name':         'Business',
        'price':        'Free',
        'cents':        '',
        'tagline':      'For professionals & labels',
        'btn_text':     'Choose Business',
        'btn_color':    '#7C3AED',
        'btn_hover':    '#6D28D9',
        'btn_text_col': '#FFFFFF',
        'card_bg':      '#0A0A1E',
        'border':       '#7C3AED',
        'glow':         False,
        'free':         True,
        'features': [
            'Unlimited songs per month',
            'Batch processing — files & folders',
            'Min 60s  /  Max 1200s duration',
            'Saved detection history',
            'File-generated spectrogram + waveform',
            'Gauge meter + dropdown with colour coding',
            'Export to CSV + PDF report',
            'Multiple agents for batch processing',
            "Colour coded indications of track authenticity in the song's timeline",
            'Smooth music player functionality with built in playhead, pause, and restart buttons'
        ],
    },
]

# Card geometry
CARD_W       = 660     # wider cards (capped to fit the screen at layout time)
CARD_PAD_X   = 50      # gap between cards
HOVER_GROW   = 8      # px to grow on each side on hover
ANIM_STEPS   = 8       # frames for hover animation
ANIM_MS      = 8     # ms per frame

# Enterprise banner (full-width gray box below the tiers)
ENT_BG       = '#2E2E38'   # gray
ENT_H        = 220         # banner height
ENT_MARGIN   = 60          # side margin so it spans nearly the full screen width


# ── Hover-animated card ───────────────────────────────────────────────────────
class PlanCard:
    """
    A pricing card placed with place() so hover scaling never shifts neighbours.
    """
    def __init__(self, parent, plan, base_x, base_y, card_w, card_h, scale=1.0,
                 is_current=False, feat_scale=1.0, on_select=None):
        self.parent   = parent
        self.plan     = plan
        self.base_x   = base_x
        self.base_y   = base_y
        self.base_w   = card_w
        self.base_h   = card_h
        self.s        = scale    # <1.0 shrinks fonts + paddings to fit the screen
        self.fs_scale = feat_scale  # extra shrink for a plan with many features
        self.is_current = is_current
        self.on_select = on_select
        self._grow    = 0.0      # 0 = resting, 1 = fully hovered
        self._target  = 0.0
        self._job     = None

        self.frame = tk.Frame(parent, bg=plan['card_bg'],
                              highlightbackground=plan['border'],
                              highlightthickness=2 if plan['glow'] else 1,
                              bd=0)
        self._place(0.0)
        self._build_content()
        if not self.is_current:
            self._bind_hover()

    # ── scale helpers ───────────────────────────────────────────────────────────
    def _f(self, size):   return max(7, int(size * self.s))   # font size
    def _p(self, v):      return max(1, int(v * self.s))      # padding / pixels

    # ── geometry ──────────────────────────────────────────────────────────────
    def _place(self, grow):
        g = int(grow * HOVER_GROW)
        self.frame.place(
            x = self.base_x - g,
            y = self.base_y - g,
            width  = self.base_w + g * 2,
            height = self.base_h + g * 2,
        )

    # ── content ───────────────────────────────────────────────────────────────
    def _build_content(self):
        p   = self.plan
        bg  = p['card_bg']
        f, pd = self._f, self._p

        # Plan name
        tk.Label(self.frame, text=p['name'], bg=bg, fg='#FFFFFF',
                 font=('Arial', f(32), 'bold'), anchor='w'
                 ).pack(anchor='w', padx=pd(38), pady=(pd(34), 0))

        # Price row
        price_row = tk.Frame(self.frame, bg=bg)
        price_row.pack(anchor='w', padx=pd(32), pady=(pd(8), 0))

        if p.get('free'):
            tk.Label(price_row, text='Free', bg=bg, fg='#FFFFFF',
                     font=('Arial', f(58), 'bold')).pack(side='left')
        else:
            tk.Label(price_row, text='$', bg=bg, fg='#FFFFFF',
                     font=('Arial', f(22), 'bold')).pack(
                         side='left', anchor='n', pady=pd(10))
            tk.Label(price_row, text=p['price'], bg=bg, fg='#FFFFFF',
                     font=('Arial', f(72), 'bold')).pack(side='left')
            sub = tk.Frame(price_row, bg=bg)
            sub.pack(side='left', anchor='s', pady=pd(16))
            tk.Label(sub, text=p['cents'], bg=bg, fg='#FFFFFF',
                     font=('Arial', f(20), 'bold')).pack(anchor='w')
            tk.Label(sub, text='USD / month', bg=bg, fg='#556677',
                     font=('Arial', f(12))).pack(anchor='w')

        # Tagline
        tk.Label(self.frame, text=p['tagline'], bg=bg, fg='#5A6A7A',
                 font=('Arial', f(13)), anchor='w'
                 ).pack(anchor='w', padx=pd(38), pady=(pd(4), pd(18)))

        # Subscribe button (use tk.Button for reliable bg support). The active
        # plan gets a non-actionable "Current Plan" button instead.
        if self.is_current:
            btn_text, btn_bg, btn_fg = 'Current Plan', '#292932', '#90909C'
        else:
            btn_text, btn_bg, btn_fg = p['btn_text'], p['btn_color'], p['btn_text_col']
        disabled = self.is_current
        btn = tk.Button(
            self.frame,
            text=btn_text,
            bg=btn_bg,
            fg=btn_fg,
            activebackground=btn_bg,
            activeforeground=btn_fg,
            font=('Arial', f(13), 'bold'),
            relief='flat', bd=0,
            cursor='arrow' if disabled else 'hand2',
            padx=0, pady=pd(14),
            state='disabled' if disabled else 'normal',
            disabledforeground=btn_fg,
            command=(lambda: self.on_select(p['name']))
                    if self.on_select and not disabled else None,
        )
        btn.pack(fill='x', padx=pd(34), pady=(0, pd(22)))
        if not disabled:
            btn.bind('<Enter>', lambda e: btn.config(bg=p['btn_hover']))
            btn.bind('<Leave>', lambda e: btn.config(bg=p['btn_color']))

        # Divider
        tk.Frame(self.frame, bg='#1E2030', height=1).pack(fill='x', padx=pd(34), pady=(0, pd(18)))

        # Features (feature font/padding can be shrunk further per-plan so a
        # long feature list still fits inside the fixed card height)
        fs   = self.fs_scale
        f_ck = max(7, int(16 * fs * self.s))
        f_ft = max(7, int(15 * fs * self.s))
        rpad = max(1, int(7 * fs * self.s))
        # Let feature text use (almost) the full card width so long features wrap
        # onto fewer lines and fit inside the card.
        wl = max(180, self.base_w - pd(104))
        for feat in p['features']:
            row = tk.Frame(self.frame, bg=bg)
            row.pack(fill='x', padx=pd(38), pady=rpad, anchor='w')
            tk.Label(row, text='✓', bg=bg, fg='#38BDF8',
                     font=('Arial', f_ck, 'bold')).pack(side='left', padx=(0, pd(12)))
            tk.Label(row, text=feat, bg=bg, fg='#AABBCC',
                     font=('Arial', f_ft), anchor='w',
                     wraplength=wl, justify='left').pack(side='left', anchor='w')

        # Bottom spacer
        tk.Frame(self.frame, bg=bg, height=pd(26)).pack()

    # ── hover animation ───────────────────────────────────────────────────────
    def _bind_hover(self):
        self.frame.bind('<Enter>', self._on_enter)
        self.frame.bind('<Leave>', self._on_leave)
        # Bind children so Enter still triggers growth
        self._bind_recursive(self.frame)

    def _bind_recursive(self, w):
        w.bind('<Enter>', self._on_enter)
        w.bind('<Leave>', self._on_leave)
        for child in w.winfo_children():
            self._bind_recursive(child, )

    def _on_enter(self, _=None):
        self._target = 1.0
        self._tick()

    def _on_leave(self, event):
    # Only shrink if cursor truly left the card's bounding box
        cx = self.frame.winfo_rootx()
        cy = self.frame.winfo_rooty()
        cw = self.frame.winfo_width()
        ch = self.frame.winfo_height()
        if cx <= event.x_root <= cx + cw and cy <= event.y_root <= cy + ch:
            return  # still inside — ignore the leave
        self._target = 0.0
        self._tick()

    def _tick(self):
        if self._job:
            self.frame.after_cancel(self._job)
        diff = self._target - self._grow
        if abs(diff) < 0.05:
            self._grow = self._target
            self._place(self._grow)
            # brighten/dim border on full hover
            self.frame.config(
                highlightbackground='#FFFFFF' if self._grow > 0.5 else self.plan['border'],
                highlightthickness=2
            )
            return
        self._grow += diff * 0.6
        self._place(self._grow)
        self._job = self.frame.after(ANIM_MS, self._tick)


# ── Contact composer ──────────────────────────────────────────────────────────
class ContactComposer:
    """Modal, Gmail-style contact form backed by the authenticated API."""

    def __init__(self, master, current_user_email, auth_token, on_close=None):
        self.master = master
        self.current_user_email = current_user_email or 'No authenticated account'
        self.auth_token = auth_token
        self.on_close = on_close
        self._sending = False
        self._message_missing = False
        self._title_missing = False
        self._notice = None

        master.update_idletasks()
        sw = max(master.winfo_screenwidth(), 900)
        sh = max(master.winfo_screenheight(), 700)
        width = min(860, int(sw * .88))
        height = min(720, int(sh * .88))
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 2)

        self.window = tk.Toplevel(master, bg='#0A0A18')
        self.window.overrideredirect(True)
        self.window.transient(master)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        self.window.minsize(680, 590)
        self.window.lift()
        self.window.grab_set()
        self.window.bind('<Escape>', lambda _e: self._cancel())

        shell = tk.Frame(self.window, bg='#0A0A18',
                         highlightbackground='#303044', highlightthickness=1)
        shell.pack(fill='both', expand=True)

        title_bar = tk.Frame(shell, bg='#181827', height=52)
        title_bar.pack(fill='x')
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text='New message', bg='#181827', fg='#FFFFFF',
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=20)
        close_btn = tk.Button(title_bar, text='×', command=self._cancel,
                              bg='#181827', fg='#C9CED8',
                              activebackground='#B3261E', activeforeground='#FFFFFF',
                              relief='flat', bd=0, font=('Segoe UI', 20),
                              width=4, cursor='hand2')
        close_btn.pack(side='right', fill='y')
        close_btn.bind('<Enter>', lambda _e: close_btn.config(bg='#B3261E', fg='#FFFFFF'))
        close_btn.bind('<Leave>', lambda _e: close_btn.config(bg='#181827', fg='#C9CED8'))
        self._make_draggable(title_bar)

        body = tk.Frame(shell, bg='#0A0A18')
        body.pack(fill='both', expand=True, padx=34, pady=(22, 24))

        self._fixed_row(body, 'To:', CONTACT_EMAIL)
        self._fixed_row(body, 'From:', self.current_user_email)
        tk.Frame(body, bg='#29293A', height=1).pack(fill='x', pady=(8, 16))

        tk.Label(body, text='Title', bg='#0A0A18', fg='#FFFFFF',
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 6))
        self.title_border = tk.Frame(body, bg=TURQUOISE)
        self.title_border.pack(fill='x')
        self.title_var = tk.StringVar()
        title_vcmd = (self.window.register(self._validate_title_length), '%P')
        self.title_entry = tk.Entry(
            self.title_border, textvariable=self.title_var,
            validate='key', validatecommand=title_vcmd,
            bg='#111120', fg='#FFFFFF', insertbackground='#FFFFFF',
            selectbackground='#245F6A', selectforeground='#FFFFFF',
            relief='flat', bd=0, font=('Segoe UI', 12))
        self.title_entry.pack(fill='x', padx=2, pady=2, ipady=10)
        self.title_var.trace_add('write', self._on_title_change)

        tk.Label(body, text='Message', bg='#0A0A18', fg='#FFFFFF',
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(18, 6))
        self.message_border = tk.Frame(body, bg=TURQUOISE)
        self.message_border.pack(fill='both', expand=True)
        message_inner = tk.Frame(self.message_border, bg='#111120')
        message_inner.pack(fill='both', expand=True, padx=2, pady=2)
        self.message_text = tk.Text(
            message_inner, bg='#111120', fg='#FFFFFF',
            insertbackground='#FFFFFF', selectbackground='#245F6A',
            selectforeground='#FFFFFF', relief='flat', bd=0,
            wrap='word', undo=True, font=('Segoe UI', 11),
            padx=12, pady=10)
        message_scroll = tk.Scrollbar(message_inner, command=self.message_text.yview,
                                      bg='#1B1B2B', troughcolor='#111120',
                                      activebackground='#40E0D0', relief='flat', bd=0)
        self.message_text.configure(yscrollcommand=message_scroll.set)
        message_scroll.pack(side='right', fill='y')
        self.message_text.pack(side='left', fill='both', expand=True)
        self.message_text.bind('<<Modified>>', self._on_message_modified)
        self.message_text.bind('<KeyPress>', self._guard_message_limit, add='+')

        counter_row = tk.Frame(body, bg='#0A0A18')
        counter_row.pack(fill='x', pady=(5, 0))
        self.status_label = tk.Label(counter_row, text='', bg='#0A0A18',
                                     fg=TURQUOISE, font=('Segoe UI', 10))
        self.status_label.pack(side='left')
        self.counter_label = tk.Label(counter_row, text='0 / 1000',
                                      bg='#0A0A18', fg='#8D95A5',
                                      font=('Consolas', 10))
        self.counter_label.pack(side='right')

        actions = tk.Frame(body, bg='#0A0A18')
        actions.pack(fill='x', pady=(18, 0))
        self.cancel_btn = tk.Button(
            actions, text='Cancel', command=self._cancel,
            bg='#252535', fg='#D5D9E2', activebackground='#343448',
            activeforeground='#FFFFFF', relief='flat', bd=0,
            font=('Segoe UI', 11, 'bold'), padx=24, pady=10,
            cursor='hand2')
        self.cancel_btn.pack(side='left')
        self.send_btn = tk.Button(
            actions, text='Send  ✈', command=self._send,
            bg='#2577E8', fg='#FFFFFF', activebackground='#3188F4',
            activeforeground='#FFFFFF', relief='flat', bd=0,
            font=('Segoe UI', 11, 'bold'), padx=28, pady=10,
            cursor='hand2')
        self.send_btn.pack(side='right')

        self.window.after(80, self.title_entry.focus_set)

    def _fixed_row(self, parent, label, value):
        row = tk.Frame(parent, bg='#0A0A18')
        row.pack(fill='x', pady=5)
        tk.Label(row, text=label, bg='#0A0A18', fg='#FFFFFF',
                 font=('Segoe UI', 11, 'bold'), width=7,
                 anchor='w').pack(side='left')
        tk.Label(row, text=value, bg='#0A0A18', fg='#72C7F4',
                 font=('Segoe UI', 11), anchor='w').pack(side='left')

    def _make_draggable(self, widget):
        drag = {'x': 0, 'y': 0}

        def start(event):
            drag['x'], drag['y'] = event.x_root, event.y_root

        def move(event):
            dx, dy = event.x_root - drag['x'], event.y_root - drag['y']
            drag['x'], drag['y'] = event.x_root, event.y_root
            self.window.geometry(f'+{self.window.winfo_x() + dx}+{self.window.winfo_y() + dy}')

        widget.bind('<ButtonPress-1>', start)
        widget.bind('<B1-Motion>', move)

    @staticmethod
    def _validate_title_length(proposed):
        return len(proposed) <= 100

    def _on_title_change(self, *_args):
        if self.title_var.get().strip():
            self._title_missing = False
        self.title_border.configure(
            bg=FIELD_ERROR if self._title_missing else TURQUOISE)

    def _guard_message_limit(self, event):
        current = self.message_text.get('1.0', 'end-1c')
        navigation = {
            'BackSpace', 'Delete', 'Left', 'Right', 'Up', 'Down',
            'Home', 'End', 'Prior', 'Next', 'Tab', 'Escape'
        }
        # Keep editing shortcuts (Ctrl+A/C/X/V/Z) functional at the limit.
        if len(current) < 1000 or event.keysym in navigation or (event.state & 0x4):
            return None
        try:
            self.message_text.index('sel.first')
            return None
        except tk.TclError:
            pass
        if event.char:
            return 'break'
        return None

    def _on_message_modified(self, _event=None):
        if not self.message_text.edit_modified():
            return
        self.message_text.edit_modified(False)
        value = self.message_text.get('1.0', 'end-1c')
        if len(value) > 1000:
            self.message_text.delete('1.0 + 1000 chars', 'end-1c')
            value = self.message_text.get('1.0', 'end-1c')
            self.message_text.edit_modified(False)
        if value.strip():
            self._message_missing = False
        count = len(value)
        self.counter_label.configure(
            text=f'{count} / 1000',
            fg=FIELD_ERROR if count >= 1000 else '#8D95A5')
        border = FIELD_ERROR if (count >= 1000 or self._message_missing) else TURQUOISE
        self.message_border.configure(bg=border)

    def _cancel(self):
        if self._sending:
            return
        self._close()

    def _close(self):
        if self._notice is not None:
            try: self._notice.destroy()
            except tk.TclError: pass
            self._notice = None
        try: self.window.grab_release()
        except tk.TclError: pass
        try: self.window.destroy()
        except tk.TclError: pass
        try:
            if self.master.winfo_exists():
                self.master.grab_set()
                self.master.lift()
        except tk.TclError:
            pass
        if self.on_close:
            self.on_close()

    def _send(self):
        if self._sending:
            return
        title = self.title_var.get().strip()
        message = self.message_text.get('1.0', 'end-1c').strip()
        self._title_missing = not bool(title)
        self._message_missing = not bool(message)
        self.title_border.configure(bg=FIELD_ERROR if self._title_missing else TURQUOISE)
        message_count = len(self.message_text.get('1.0', 'end-1c'))
        self.message_border.configure(
            bg=FIELD_ERROR if (self._message_missing or message_count >= 1000) else TURQUOISE)

        if self._title_missing or self._message_missing:
            self._show_warning(
                'Transmission payload rejected: required fields “Title” and '
                '“Message” must each contain non-whitespace data before dispatch.')
            return
        if not self.auth_token:
            self._show_warning(
                'Authentication context unavailable: the message cannot be '
                'dispatched without a valid signed-in session token.')
            return

        self._sending = True
        self.send_btn.configure(text='Transmitting…', state='disabled', cursor='watch')
        self.cancel_btn.configure(state='disabled', cursor='watch')
        self.status_label.configure(text='Establishing authenticated transport…')
        result = {}

        def request_worker():
            try:
                response = requests.post(
                    f'{SERVER_URL}/contact',
                    json={'title': title, 'message': message},
                    headers={'Authorization': f'Bearer {self.auth_token}'},
                    timeout=20)
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if response.ok:
                    result.update(done=True, ok=True)
                else:
                    result.update(done=True, ok=False,
                                  error=payload.get('detail') or
                                        f'Server returned HTTP {response.status_code}.')
            except requests.RequestException as exc:
                result.update(done=True, ok=False,
                              error=f'Contact service unavailable: {exc}')

        threading.Thread(target=request_worker, daemon=True).start()

        def poll_result():
            try:
                exists = self.window.winfo_exists()
            except tk.TclError:
                exists = False
            if not exists:
                return
            if not result.get('done'):
                self.window.after(80, poll_result)
                return
            if result.get('ok'):
                self.status_label.configure(text='Message transmitted successfully.', fg='#41E37C')
                self.send_btn.configure(text='Sent  ✓', bg='#176B38')
                self.window.after(900, self._finish_success)
            else:
                self._sending = False
                self.send_btn.configure(text='Send  ✈', state='normal', cursor='hand2')
                self.cancel_btn.configure(state='normal', cursor='hand2')
                self.status_label.configure(text='Transmission failed.', fg=FIELD_ERROR)
                self._show_warning(str(result.get('error', 'Unknown transport failure.')))

        self.window.after(80, poll_result)

    def _finish_success(self):
        self._sending = False
        self._close()

    def _show_warning(self, message):
        if self._notice is not None:
            try:
                self._notice.lift()
                return
            except tk.TclError:
                self._notice = None

        self.window.update_idletasks()
        width, height = 640, 300
        x = self.window.winfo_rootx() + (self.window.winfo_width() - width) // 2
        y = self.window.winfo_rooty() + (self.window.winfo_height() - height) // 2
        notice = tk.Toplevel(self.window, bg='#11111D')
        notice.overrideredirect(True)
        notice.transient(self.window)
        notice.geometry(f'{width}x{height}+{x}+{y}')
        notice.lift()
        notice.grab_set()
        self._notice = notice

        shell = tk.Frame(notice, bg='#11111D',
                         highlightbackground='#4A4A58', highlightthickness=1)
        shell.pack(fill='both', expand=True)
        bar = tk.Frame(shell, bg='#20202D', height=46)
        bar.pack(fill='x')
        bar.pack_propagate(False)
        tk.Label(bar, text='Input Contract Violation', bg='#20202D', fg='#FFFFFF',
                 font=('Segoe UI', 12, 'bold')).pack(side='left', padx=16)

        def dismiss():
            try: notice.grab_release()
            except tk.TclError: pass
            try: notice.destroy()
            except tk.TclError: pass
            self._notice = None
            try:
                self.window.grab_set()
                self.window.lift()
            except tk.TclError:
                pass

        close_btn = tk.Button(bar, text='×', command=dismiss,
                              bg='#20202D', fg='#D6DAE2',
                              activebackground='#B3261E', activeforeground='#FFFFFF',
                              relief='flat', bd=0, font=('Segoe UI', 18), width=4)
        close_btn.pack(side='right', fill='y')
        notice.bind('<Escape>', lambda _e: dismiss())

        content = tk.Frame(shell, bg='#11111D')
        content.pack(fill='both', expand=True, padx=28, pady=(24, 12))
        tk.Label(content, text=message, bg='#11111D', fg='#E5E7EC',
                 font=('Segoe UI', 11), justify='left', anchor='w',
                 wraplength=430).pack(side='left', fill='both', expand=True)
        icon = tk.Canvas(content, width=105, height=105, bg='#11111D',
                         highlightthickness=0)
        icon.pack(side='right', padx=(18, 0))
        icon.create_polygon(52, 7, 99, 94, 5, 94,
                            fill='#F5C542', outline='#FFD95C', width=2)
        icon.create_text(52, 65, text='!', fill='#181818',
                         font=('Segoe UI', 40, 'bold'))

        buttons = tk.Frame(shell, bg='#11111D')
        buttons.pack(fill='x', padx=24, pady=(0, 20))
        tk.Button(buttons, text='OK', command=dismiss,
                  bg='#2F7BEE', fg='#FFFFFF', activebackground='#3B88FA',
                  activeforeground='#FFFFFF', relief='flat', bd=0,
                  font=('Segoe UI', 10, 'bold'), padx=28, pady=8,
                  cursor='hand2').pack(side='right')
        notice.after(50, notice.focus_force)


# ── Upgrade view (embeddable into any parent widget) ──────────────────────────
class UpgradePlanView:
    """
    Builds the full upgrade-plan UI into an arbitrary parent widget, so it can be
    shown standalone (UpgradePlanApp) or overlaid inside another window (e.g. the
    main detector GUI). Pass ``on_back`` to get a top-left ← button that calls it.
    """
    def __init__(self, master, sw, sh, on_back=None, current_plan=None,
                 on_plan_selected=None, current_user_email=None, auth_token=None):
        self.master   = master
        self._sw      = sw
        self._sh      = sh
        self._on_back = on_back
        self._current = current_plan   # active plan name, e.g. 'Premium'
        self._on_plan_selected = on_plan_selected
        self._current_user_email = current_user_email
        self._auth_token = auth_token
        self._contact_composer = None
        self._selection_locked = False
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        sw, sh = self._sw, self._sh

        # ── Uniform scale so the whole page (title + 3 cards + enterprise banner)
        # fits the window height without scrolling. Derived from the tallest plan.
        max_feats = max(len(p['features']) for p in PLANS)
        # Extra per-feature height so the longest plan (Business, with several
        # wrapping features) isn't clipped at the bottom of its card.
        natural_card_h = 480 + max_feats * 56
        natural_total  = 290 + natural_card_h + 90 + ENT_H + 90
        scale = min(1.0, (sh - 40) / natural_total)
        scale = max(0.5, scale)
        self.scale = scale
        f  = lambda s: max(7, int(s * scale))
        pd = lambda v: max(1, int(v * scale))

        # ── Card cluster geometry. Cap each card at its design width so the row
        # stays a sensible size; the cluster is then left-aligned with a small
        # margin and the title is centred over it.
        n         = len(PLANS)
        gap       = int(CARD_PAD_X * scale)
        max_by_w  = (sw - 2 * ENT_MARGIN - (n - 1) * gap) // n
        card_w    = max(200, min(int(CARD_W * scale), max_by_w))
        cluster_w = n * card_w + (n - 1) * gap
        self._card_w    = card_w
        self._card_gap  = gap
        self._card_h    = int(natural_card_h * scale)

        # Background canvas
        self.bg_canvas = tk.Canvas(self.master, width=sw, height=sh,
                                bg='#040414', highlightthickness=0)
        self.bg_canvas.place(x=0, y=0)
        bg_img = make_gradient(sw, sh)
        self._bg_tk = ImageTk.PhotoImage(bg_img)
        self.bg_canvas.create_image(0, 0, anchor='nw', image=self._bg_tk)

        # Scrollable overlay — placed with relwidth/relheight so it matches the
        # REAL viewport. (CTk scales an explicit width= by its own DPI factor,
        # unlike the plain-tk canvas above — that mismatch is what pushed the
        # whole page off to the right.)
        self.scroll = ctk.CTkScrollableFrame(
            self.master, fg_color='#040414',
            scrollbar_button_color='#1A2030',
            scrollbar_button_hover_color='#2A3040')
        self.scroll.place(x=0, y=0, relwidth=1.0, relheight=1.0)

        # ── Title: centred over the cluster ─────────────────────────────────────
        # Extra top padding shifts the whole page down; the taller wrap keeps the
        # "Authenticate your music" subtitle from being clipped.
        title_wrap = tk.Frame(self.scroll, bg='#040414',
                              width=cluster_w, height=int(200 * scale))
        title_wrap.pack(pady=(pd(58), pd(16)))
        title_wrap.pack_propagate(False)
        tk.Label(title_wrap, text='Upgrade Your Plan', bg='#040414', fg='#FFFFFF',
                 font=('Arial', f(52), 'bold')).pack()
        tk.Label(title_wrap, text='Authenticate your music', bg='#040414',
                 fg='#3A4A5A', font=('Arial', f(20))).pack()

        # ── Cards container (fixed cluster width + hover breathing room, centred).
        # The extra HOVER_GROW on each side keeps a card's edge from being
        # clipped when it grows on hover.
        self._container = tk.Frame(self.scroll, bg='#040414',
                                   width=cluster_w + HOVER_GROW * 2,
                                   height=self._card_h + HOVER_GROW * 2 + pd(16))
        self._container.pack(pady=(0, pd(24)))
        self._container.pack_propagate(False)

        # ── Enterprise banner (same cluster width, centred). Extra top gap so it
        # sits a bit lower, clear of the (taller) plan cards.
        ent = tk.Frame(self.scroll, bg=ENT_BG, width=cluster_w,
                       height=int(ENT_H * scale))
        ent.pack(pady=(pd(40), pd(50)))
        ent.pack_propagate(False)

        # Button first (side right) so it reserves the right edge...
        ent_btn = tk.Button(ent, text='Contact Me',
                            bg='#FFFFFF', fg='#1A1A1A',
                            activebackground='#E5E5E5', activeforeground='#1A1A1A',
                            font=('Arial', f(15), 'bold'), relief='flat', bd=0,
                            padx=pd(36), pady=pd(16), cursor='hand2',
                            command=self._open_contact)
        ent_btn.pack(side='right', padx=pd(48))
        ent_btn.bind('<Enter>', lambda e: ent_btn.config(bg='#E5E5E5'))
        ent_btn.bind('<Leave>', lambda e: ent_btn.config(bg='#FFFFFF'))

        # If Enterprise is the active plan, its button becomes "Current Plan".
        if self._current == 'Enterprise':
            ent_btn.config(text='Current Plan', state='disabled',
                           disabledforeground='#5EDD5E', bg='#173417',
                           cursor='arrow')
            ent_btn.unbind('<Enter>'); ent_btn.unbind('<Leave>')

        # ...then the title + description fill the rest on the left.
        ent_left = tk.Frame(ent, bg=ENT_BG)
        ent_left.pack(side='left', fill='both', expand=True, padx=pd(48), pady=pd(24))
        tk.Label(ent_left, text='Enterprise — Contact Us', bg=ENT_BG, fg='#FFFFFF',
                 font=('Arial', f(40), 'bold'), anchor='w').pack(anchor='w')
        tk.Label(ent_left,
                 text='REST API access with high-throughput requests · automated bulk '
                      '& folder screening at scale · webhook integration into your '
                      'ingestion pipeline · team seats with SSO · priority SLA with a '
                      'dedicated account manager · on-premise / private-cloud '
                      'deployment with data residency · white-labelled PDF reports · '
                      'per-generator attribution (Suno, Udio & more).',
                 bg=ENT_BG, fg='#FFFFFF', font=('Arial', f(15)),
                 anchor='w', justify='left', wraplength=pd(1300)
                 ).pack(anchor='w', pady=(pd(12), 0))

        # ── Back arrow (only when embedded with a callback) ─────────────────────
        # White ← in the top-left; a small gray box appears on hover; click
        # returns to whatever opened this view.
        if self._on_back is not None:
            back = tk.Button(self.master, text='←', bg='#040414', fg='#FFFFFF',
                             activebackground='#2A2A32', activeforeground='#FFFFFF',
                             relief='flat', bd=0, font=('Arial', 26, 'bold'),
                             cursor='hand2', padx=12, pady=2, command=self._on_back)
            back.place(x=22, y=18)
            back.lift()
            back.bind('<Enter>', lambda e: back.config(bg='#2A2A32'))
            back.bind('<Leave>', lambda e: back.config(bg='#040414'))

        # Defer card placement until layout resolves
        self.master.after(150, self._place_cards)

    def _open_contact(self):
        if self._contact_composer is not None:
            try:
                self._contact_composer.window.lift()
                return
            except tk.TclError:
                self._contact_composer = None

        def cleared():
            self._contact_composer = None

        self._contact_composer = ContactComposer(
            self.master, self._current_user_email, self._auth_token,
            on_close=cleared)

    def _select_plan(self, plan_name):
        """Accept exactly one plan click until the detector finishes rebuilding."""
        if self._selection_locked or plan_name == self._current:
            return
        self._selection_locked = True

        # Keep the plan page visibly in place. Disable every button recursively
        # (cards, Back, and Contact Me), then add only a compact status badge;
        # this prevents duplicate clicks without replacing the chooser screen.
        def disable_buttons(widget):
            if isinstance(widget, tk.Button):
                try: widget.configure(state='disabled', cursor='watch')
                except tk.TclError: pass
            for child in widget.winfo_children():
                disable_buttons(child)
        disable_buttons(self.master)

        status = tk.Label(
            self.master, text=f'Applying {plan_name} plan…',
            bg='#24242E', fg='#C2C7D0', relief='flat', bd=0,
            font=('Segoe UI', 13, 'bold'), padx=18, pady=8,
            cursor='watch')
        status.place(relx=.5, y=20, anchor='n')
        status.lift()

        if self._on_plan_selected is not None:
            self.master.after_idle(
                lambda: self._on_plan_selected(plan_name))

    def _place_cards(self):
        # The container is exactly the cluster width and left-aligned, so cards
        # sit at x = 0, card_w+gap, 2*(card_w+gap) inside it.
        card_w = self._card_w
        gap    = self._card_gap
        max_feats = max(len(p['features']) for p in PLANS)
        self._cards = []
        for i, plan in enumerate(PLANS):
            x = HOVER_GROW + i * (card_w + gap)
            y = HOVER_GROW
            # Plans with a long feature list get a smaller feature font so every
            # feature fits inside the fixed card height.
            feat_scale = 0.8 if len(plan['features']) >= max_feats else 1.0
            card = PlanCard(self._container, plan, x, y, card_w, self._card_h,
                            scale=getattr(self, 'scale', 1.0),
                            is_current=(plan['name'] == self._current),
                            feat_scale=feat_scale,
                            on_select=self._select_plan)
            self._cards.append(card)


# ── Standalone app ─────────────────────────────────────────────────────────────
class UpgradePlanApp:
    def __init__(self):
        ctk.set_appearance_mode('dark')
        self.root = ctk.CTk()
        self.root.title('AI Music Detector — Upgrade')

        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'{sw}x{sh}')
        self.root.state('zoomed')
        self.root.resizable(True, True)

        self.view = UpgradePlanView(self.root, sw, sh)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = UpgradePlanApp()
    app.run()
