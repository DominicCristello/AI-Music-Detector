import customtkinter as ctk
import tkinter as tk
import threading
import requests
import time
import ctypes
import os
import re
import sys
import subprocess
import atexit
from urllib.parse import urlparse

# Keep one logical UI scale for the lifetime of the process.  CustomTkinter's
# default per-monitor watcher rescales CTk widgets and fonts whenever a window
# crosses onto a monitor with a different Windows scaling percentage, while
# the plain Tk widgets around them do not follow exactly the same rules.  That
# mismatch is what made button text and padding jump out of alignment.
try:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_widget_scaling(1.0)
    ctk.set_window_scaling(1.0)
except Exception:
    pass

try:
    # System-DPI awareness keeps Tk's layout metrics fixed between monitors.
    # Windows may scale the completed window as a single surface on another
    # monitor, preserving the proportions of fonts, buttons, and fields.
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
# Where your FastAPI auth server is running. Localhost during development;
# swap for your hosted URL (https://...) once deployed.
SERVER_URL = os.environ.get(
    "AI_DETECTOR_SERVER_URL", "http://localhost:8000").rstrip('/')

# Basic client-side email sanity check so obviously-invalid input is caught
# with a friendly message instead of a server 422 validation error.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _looks_like_email(s: str) -> bool:
    return bool(_EMAIL_RE.match(s.strip()))

# Background colour — matched to upgrade_plan.py's background canvas (#040414).
BG_MAIN   = '#040414'
BG_CARD   = '#0B0B1A'
WHITE     = '#FFFFFF'
BLACK     = '#000000'
BLUE      = '#38BDF8'
BLUE_H    = '#0EA5E9'
GRAY      = '#6A7A8A'
TEXT_DIM  = '#6A9AB0'
PURPLE    = '#7C3AED'      # back-button hover background
ERR_BG    = '#3A1518'      # light-red error box fill
ERR_TEXT  = '#FF8080'      # error text colour
OK_BG     = '#12331C'      # green success box fill
OK_TEXT   = '#5EE08A'      # green success text + checkmark
OK_BORDER = '#1F5A33'      # green success box border

FIELD_W = 520


# ══════════════════════════════════════════════════════════════════════════════
#  Hover-styled button helper
# ══════════════════════════════════════════════════════════════════════════════
def make_button(parent, text, command, base_bg, base_fg, hover_bg, hover_fg,
                width_pad=40, font_size=15, pad_y=9):
    """A tk.Button with custom hover colours."""
    btn = tk.Button(parent, text=text, command=command,
                    bg=base_bg, fg=base_fg,
                    activebackground=hover_bg, activeforeground=hover_fg,
                    relief='solid', bd=2, font=('Segoe UI', font_size, 'bold'),
                    cursor='hand2', padx=width_pad, pady=pad_y)
    btn.bind('<Enter>', lambda e: btn.configure(bg=hover_bg, fg=hover_fg))
    btn.bind('<Leave>', lambda e: btn.configure(bg=base_bg, fg=base_fg))
    return btn


# ── Time formatting: seconds if < 61s, else minutes ──────────────────────────
def _fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 61:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    mins = seconds // 60
    rem = seconds % 60
    if rem == 0:
        return f"{mins} minute{'s' if mins != 1 else ''}"
    return (f"{mins} minute{'s' if mins != 1 else ''} "
            f"{rem} second{'s' if rem != 1 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
#  Login / Sign-up / Forgot-password screen (multi-screen navigation)
# ══════════════════════════════════════════════════════════════════════════════
class LoginScreen:
    """
    Multi-screen auth UI:
      - landing : title + 'Sign up' and 'Login' buttons
      - signup  : email + password + 'Create Account'
      - login   : email + password + 'Login' + 'Forgot Password' link
      - forgot  : email -> send code -> verify code -> set new password
    On successful login, calls on_success(token, tier, email).
    """
    def __init__(self, on_success):
        ctk.set_appearance_mode('dark')
        self.on_success = on_success

        # Boot the auth backend as early as possible so it's ready by the time
        # the user submits credentials.
        self._server_proc = None
        self._start_backend()

        # Pre-import the (heavy) startup app in the background while the user is
        # on the login screen. appGUI_Startup pulls in librosa + the whole main
        # app, which is a slow first-time import — doing it now means the handoff
        # after login is instant instead of a blank few-second gap. Only warms
        # sys.modules (no Tk objects are created at import time), so it's safe
        # off the main thread.
        self._startup_ready = threading.Event()

        def _preload_startup():
            try:
                import appGUI_Startup  # noqa: F401 — just warming the import cache
            except Exception as e:
                print(f"[LoginScreen] startup preload failed (will retry on "
                      f"login): {e}")
            finally:
                self._startup_ready.set()

        threading.Thread(target=_preload_startup, daemon=True).start()

        self.root = ctk.CTk()
        self.root.title("AI Music Detector — Sign up / Login")
        self.root.configure(fg_color=BG_MAIN)

        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # Floor so minimizing/restoring never shrinks the window to a tiny box,
        # and a comfortable centred size to restore to when un-maximized.
        self.root.minsize(int(sw * 0.55), int(sh * 0.60))
        rw, rh = int(sw * 0.75), int(sh * 0.80)
        self.root.geometry(f"{rw}x{rh}+{(sw - rw) // 2}+{(sh - rh) // 2}")
        self.root.after(0, lambda: self.root.state('zoomed'))

        # DPI scaling factor. CTk scales widget sizes (e.g. CTkEntry width) by
        # this, but plain tk.Buttons placed with x=... are NOT scaled — so we
        # scale button offsets ourselves to keep them aligned with the fields.
        try:
            self._ui_scale = ctk.ScalingTracker.get_window_scaling(self.root)
        except Exception:
            self._ui_scale = 1.0

        self._busy = False
        self._handoff = None

        # ── Login-failure lockout (5 fails -> 30s, 60s, 90s, ...) ────────────
        self._login_fails = 0
        self._login_lock_count = 0
        self._login_locked = False
        self._login_lock_remaining = 0
        self._login_lock_job = None

        # ── Forgot-password attempt lockout (3 fails -> 30s, 60s, ...) ───────
        self._forgot_fails = 0
        self._forgot_lock_count = 0
        self._forgot_locked_until = 0.0     # epoch seconds; 0 = not locked

        self._screen_widgets = []
        self._error_box = None
        self._reset_email = None
        self._reset_code = None

        self._show_landing()

    # ══════════════════════════════════════════════════════════════════════════
    #  Screen management
    # ══════════════════════════════════════════════════════════════════════════
    def _clear_screen(self):
        for w in self._screen_widgets:
            try: w.destroy()
            except Exception: pass
        self._screen_widgets = []
        self._hide_error()

    def _add(self, widget):
        self._screen_widgets.append(widget)
        return widget

    def _title(self, text, rely=0.10, size=88):
        lbl = ctk.CTkLabel(self.root, text=text,
                           font=ctk.CTkFont('Segoe UI', size, 'bold'),
                           text_color=WHITE, fg_color=BG_MAIN)
        lbl.place(relx=0.5, rely=rely, anchor='center')
        return self._add(lbl)

    # ── Unified notification box: horizontally centred, at the bottom ───────
    def _show_notice(self, message, icon, bg, fg, border, auto_ms=None):
        self._hide_error()
        self._error_box = tk.Frame(self.root, bg=bg, bd=0,
                                   highlightbackground=border,
                                   highlightthickness=1)
        tk.Label(self._error_box, text=icon, bg=bg, fg=fg,
                 font=('Segoe UI', 14, 'bold')).pack(side='left',
                                                     padx=(14, 8), pady=10)
        tk.Label(self._error_box, text=message, bg=bg, fg=fg,
                 font=('Segoe UI', 13)).pack(side='left', padx=(0, 16), pady=10)
        self._error_box.place(relx=0.5, rely=0.92, anchor='center')
        # Optional auto-dismiss (e.g. success messages fade after a minute).
        if auto_ms:
            self._notice_job = self.root.after(auto_ms, self._hide_error)

    def _show_error(self, message):
        self._show_notice(message, '⚠', ERR_BG, ERR_TEXT, '#5A2025')

    def _show_success(self, message):
        # Green, checkmark, and clears itself after 60s of sitting there.
        self._show_notice(message, '✓', OK_BG, OK_TEXT, OK_BORDER,
                          auto_ms=60_000)

    def _hide_error(self):
        job = getattr(self, '_notice_job', None)
        if job is not None:
            try: self.root.after_cancel(job)
            except Exception: pass
            self._notice_job = None
        if self._error_box is not None:
            try: self._error_box.destroy()
            except Exception: pass
            self._error_box = None

    # ══════════════════════════════════════════════════════════════════════════
    #  Landing screen
    # ══════════════════════════════════════════════════════════════════════════
    def _show_landing(self):
        self._clear_screen()
        self._title("Sign Up / Login", rely=0.20, size=88)

        signup_btn = make_button(
            self.root, "Sign up", self._show_signup,
            base_bg=BG_MAIN, base_fg=WHITE, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=70, font_size=24, pad_y=14)
        signup_btn.place(relx=0.5, rely=0.52, anchor='center')
        self._add(signup_btn)

        login_btn = make_button(
            self.root, "Login", self._show_login,
            base_bg=BG_MAIN, base_fg=BLUE, hover_bg=BLUE, hover_fg=BLACK,
            width_pad=84, font_size=24, pad_y=14)
        login_btn.place(relx=0.5, rely=0.64, anchor='center')
        self._add(login_btn)

    # ══════════════════════════════════════════════════════════════════════════
    #  Shared: email + password card
    # ══════════════════════════════════════════════════════════════════════════
    def _build_credentials_card(self, signup=False):
        card = ctk.CTkFrame(self.root, fg_color=BG_MAIN,
                            border_width=0,
                            corner_radius=0, width=FIELD_W + 40, height=240)
        card.place(relx=0.5, rely=0.44, anchor='center')
        card.pack_propagate(False)
        self._add(card)

        email = ctk.CTkEntry(card, placeholder_text="Email",
                             width=FIELD_W, height=64, corner_radius=12,
                             fg_color=BG_CARD, border_color='#2A2A40',
                             border_width=1, text_color=WHITE,
                             font=ctk.CTkFont('Segoe UI', 22))
        email.pack(pady=(20, 22))
        # The email field stays its default grey no matter what is typed — its
        # validity is checked on submit (and ultimately by the emailed code).

        password = ctk.CTkEntry(card, placeholder_text="Password", show='•',
                                width=FIELD_W, height=64, corner_radius=12,
                                fg_color=BG_CARD, border_color='#2A2A40',
                                border_width=1, text_color=WHITE,
                                font=ctk.CTkFont('Segoe UI', 22))
        password.pack(pady=(0, 14))

        if signup:
            # Hint stays HIDDEN until the password is non-empty & invalid (red).
            self._pw_hint = ctk.CTkLabel(
                card,
                text=("Must be at least 10 characters and include an uppercase "
                      "letter, a lowercase letter, a number and a symbol."),
                text_color=ERR_TEXT, fg_color=BG_MAIN, justify='left',
                wraplength=FIELD_W, font=ctk.CTkFont('Segoe UI', 12))
            # Live: grey while empty, red while the typed password is invalid,
            # back to grey when valid — re-checked on every keystroke.
            password.bind('<KeyRelease>',
                          lambda e: self._flag_password_live(password))
        else:
            self._pw_hint = None

        return card, email, password

    def _password_ok(self, pw: str) -> bool:
        """True only if pw is 10+ chars with upper, lower, digit and symbol."""
        return bool(len(pw) >= 10
                    and re.search(r'[A-Z]', pw)
                    and re.search(r'[a-z]', pw)
                    and re.search(r'\d', pw)
                    and re.search(r'[^A-Za-z0-9]', pw))

    def _set_border(self, entry, ok):
        """Grey border when ok=True, red border when ok=False."""
        try:
            if ok:
                entry.configure(border_color='#2A2A40', border_width=1)
            else:
                entry.configure(border_color=ERR_TEXT, border_width=2)
        except Exception:
            pass

    def _flag_password_live(self, entry):
        """Grey while empty or valid; red once the typed password is invalid.
        The requirements hint is shown only while the field is red."""
        val = entry.get()
        invalid = (val != '' and not self._password_ok(val))
        self._set_border(entry, not invalid)
        hint = getattr(self, '_pw_hint', None)
        if hint is not None:
            if invalid:
                hint.pack(anchor='w', padx=20, pady=(0, 0))   # reveal
            else:
                hint.pack_forget()                            # hide again

    def _back_link(self, rely=0.82):
        back = make_button(
            self.root, "Back", self._show_landing,
            base_bg=BLUE, base_fg=WHITE, hover_bg=PURPLE, hover_fg=BLACK,
            width_pad=40, font_size=20, pad_y=10)
        back.place(relx=0.5, rely=rely, anchor='center')
        self._add(back)

    # ══════════════════════════════════════════════════════════════════════════
    #  Sign up screen
    # ══════════════════════════════════════════════════════════════════════════
    def _show_signup(self):
        self._clear_screen()
        self._title("Sign Up", rely=0.14, size=84)
        card, self._su_email, self._su_password = \
            self._build_credentials_card(signup=True)

        # Create Account button — below the password field + hint, aligned to
        # its RIGHT edge. Field right edge sits at screen-centre + FIELD_W/2.
        create_btn = make_button(
            self.root, "Create Account", self._do_signup,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=14, font_size=15, pad_y=6)
        create_btn.place(relx=0.5, rely=0.60, anchor='e',
                         x=int(FIELD_W / 2 * self._ui_scale))
        self._add(create_btn)

        self._su_password.bind('<Return>', lambda e: self._do_signup())
        self._back_link()

    def _do_signup(self):
        if self._busy:
            return
        email = self._su_email.get().strip()
        password = self._su_password.get()
        # Password gates submission (it turns red live). Email never turns red;
        # we still confirm it looks like an address before sending.
        if not self._password_ok(password):
            self._set_border(self._su_password, False)
            self._show_error("Please verify all fields before submitting.")
            return
        if not _looks_like_email(email):
            self._show_error("Please enter a valid email address.")
            return
        # /signup no longer creates the account — it emails a verification code.
        # On success we move to the code screen; the account is created only
        # once /signup/verify confirms the code.
        self._run_async('/signup', {"email": email, "password": password},
                        on_ok=lambda data: self._signup_code_sent(email, password),
                        on_fail=self._signup_failed)

    def _signup_failed(self, msg):
        try:
            self._su_email.delete(0, 'end')
            self._su_password.delete(0, 'end')
        except Exception:
            pass
        if 'exist' in msg.lower() or 'already' in msg.lower():
            msg = "Email already in use. Please try a different email."
        self._show_error(msg)

    # ── Signup step 2: email verification code ──────────────────────────────
    def _signup_code_sent(self, email, password):
        self._signup_email = email
        self._signup_password = password       # kept only for "Resend code"
        self._signup_code_fails = 0
        self._show_signup_code()

    def _show_signup_code(self):
        self._clear_screen()
        self._title("Sign Up", rely=0.14, size=84)

        # Label above the field, styled like the Email/Password fields.
        label = ctk.CTkLabel(self.root, text="6-Digit Code",
                             text_color=WHITE, fg_color=BG_MAIN,
                             font=ctk.CTkFont('Segoe UI', 22))
        label.place(relx=0.5, rely=0.37, anchor='center')
        self._add(label)

        self._sc_field = ctk.CTkEntry(
            self.root, placeholder_text="e.g. XXXXXX",
            width=FIELD_W, height=64, corner_radius=12,
            fg_color=BG_CARD, border_color='#2A2A40', border_width=1,
            text_color=WHITE, font=ctk.CTkFont('Segoe UI', 22))
        self._sc_field.place(relx=0.5, rely=0.44, anchor='center')
        self._add(self._sc_field)

        note = ctk.CTkLabel(
            self.root,
            text=f"We sent a code to {self._signup_email}. It expires in 2 minutes.",
            font=ctk.CTkFont('Segoe UI', 12), text_color=GRAY, fg_color=BG_MAIN)
        note.place(relx=0.5, rely=0.50, anchor='center')
        self._add(note)

        # "Enter" — right edge aligned to the field, same style/hover as
        # "Create Account".
        enter_btn = make_button(
            self.root, "Enter", self._do_signup_verify,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=14, font_size=15, pad_y=6)
        enter_btn.place(relx=0.5, rely=0.565, anchor='e',
                        x=int(FIELD_W / 2 * self._ui_scale))
        self._add(enter_btn)

        # "Resend Code" — left edge aligned to the field.
        resend_btn = make_button(
            self.root, "Resend Code", self._do_resend_code,
            base_bg=BG_MAIN, base_fg=BLUE, hover_bg=BLUE, hover_fg=BLACK,
            width_pad=14, font_size=15, pad_y=6)
        resend_btn.place(relx=0.5, rely=0.565, anchor='w',
                         x=int(-FIELD_W / 2 * self._ui_scale))
        self._add(resend_btn)

        # Typing again clears the red error state back to grey.
        self._sc_field.bind(
            '<KeyRelease>',
            lambda e: self._sc_field.configure(border_color='#2A2A40',
                                               border_width=1))
        self._sc_field.bind('<Return>', lambda e: self._do_signup_verify())
        self._back_link()

    def _do_signup_verify(self):
        if self._busy:
            return
        code = self._sc_field.get().strip()
        if not code:
            self._show_error("Enter the six-digit code.")
            return
        self._run_async('/signup/verify',
                        {"email": self._signup_email, "code": code},
                        on_ok=self._signup_complete,
                        on_fail=self._signup_verify_failed)

    def _signup_verify_failed(self, msg):
        self._signup_code_fails += 1
        # Clear the field (placeholder reappears) and flag it red.
        try:
            self._sc_field.delete(0, 'end')
            self._sc_field.configure(border_color=ERR_TEXT, border_width=2)
        except Exception:
            pass
        # 6th failed attempt -> the "too many attempts" screen.
        if self._signup_code_fails >= 6:
            self._show_too_many_attempts()
            return
        low = (msg or "").lower()
        if 'expired' in low:
            self._show_error("Expired Code.")
        elif 'too many' in low:
            self._show_too_many_attempts()
        else:
            self._show_error("Invalid code.")

    def _do_resend_code(self):
        if self._busy:
            return
        self._run_async('/signup',
                        {"email": self._signup_email,
                         "password": self._signup_password},
                        on_ok=lambda data: self._resend_ok(),
                        on_fail=lambda msg: self._show_error(msg))

    def _resend_ok(self):
        self._signup_code_fails = 0
        try:
            self._sc_field.delete(0, 'end')
            self._sc_field.configure(border_color='#2A2A40', border_width=1)
        except Exception:
            pass
        self._show_error("A new code has been sent to your email.")

    def _signup_complete(self, data):
        # Account is created now — send them back to the landing screen to log
        # in (login then hands off to the main app).
        self._show_landing()
        self._show_success("Account created! You can now log in.")

    def _show_too_many_attempts(self):
        self._clear_screen()
        big = ctk.CTkLabel(
            self.root, text="404 Error. Too many attempts reached",
            font=ctk.CTkFont('Segoe UI', 44, 'bold'),
            text_color=WHITE, fg_color=BG_MAIN, justify='center', wraplength=1200)
        big.place(relx=0.5, rely=0.44, anchor='center')
        self._add(big)
        small = ctk.CTkLabel(
            self.root, text="Try restarting the login sequence again",
            font=ctk.CTkFont('Segoe UI', 20), text_color=GRAY, fg_color=BG_MAIN)
        small.place(relx=0.5, rely=0.53, anchor='center')
        self._add(small)
        self._back_link()

    # ══════════════════════════════════════════════════════════════════════════
    #  Login screen
    # ══════════════════════════════════════════════════════════════════════════
    def _show_login(self):
        self._clear_screen()
        self._title("Login", rely=0.14, size=84)
        card, self._li_email, self._li_password = self._build_credentials_card()

        # Login button — below the password field, aligned to its RIGHT edge.
        self._login_btn = make_button(
            self.root, "Login", self._do_login,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=48, font_size=20, pad_y=10)
        self._login_btn.place(relx=0.5, rely=0.60, anchor='e',
                              x=int(FIELD_W / 2 * self._ui_scale))
        self._add(self._login_btn)

        # Forgot Password — below the password field, aligned to its LEFT edge.
        forgot = ctk.CTkLabel(self.root, text="Forgot Password",
                              text_color=GRAY, cursor='hand2',
                              font=ctk.CTkFont('Segoe UI', 20), fg_color=BG_MAIN)
        forgot.place(relx=0.5, rely=0.60, anchor='w', x=-FIELD_W // 2)
        forgot.bind('<Enter>', lambda e: forgot.configure(text_color=BLUE))
        forgot.bind('<Leave>', lambda e: forgot.configure(text_color=GRAY))
        forgot.bind('<Button-1>', lambda e: self._forgot_clicked())
        self._add(forgot)

        self._li_password.bind('<Return>', lambda e: self._do_login())
        self._back_link()

        if self._login_locked:
            self._login_btn.configure(state='disabled')

    def _do_login(self):
        if self._busy or self._login_locked:
            return
        email = self._li_email.get().strip()
        password = self._li_password.get()
        if not email or not password:
            self._show_error("Enter an email and password.")
            return
        if not _looks_like_email(email):
            self._show_error("Please enter a valid email address.")
            return
        self._run_async('/login', {"email": email, "password": password},
                        on_ok=self._auth_success, on_fail=self._login_failed)

    def _login_failed(self, msg):
        self._login_fails += 1
        if self._login_fails >= 5:
            self._login_fails = 0
            self._login_lock_count += 1
            self._start_login_lockout(30 * self._login_lock_count)
            return
        remaining = 5 - self._login_fails
        self._show_error(
            f"Incorrect credentials. Please try again. "
            f"({remaining} attempt{'s' if remaining != 1 else ''} left)")

    def _start_login_lockout(self, seconds):
        self._login_locked = True
        self._login_lock_remaining = seconds
        try:
            self._login_btn.configure(state='disabled')
        except Exception:
            pass
        self._tick_login_lockout()

    def _tick_login_lockout(self):
        if self._login_lock_remaining > 0:
            self._show_error(
                f"Too many failed attempts. "
                f"Try again in {_fmt_time(self._login_lock_remaining)}.")
            self._login_lock_remaining -= 1
            self._login_lock_job = self.root.after(1000,
                                                   self._tick_login_lockout)
        else:
            self._login_locked = False
            self._login_lock_job = None
            try:
                self._login_btn.configure(state='normal')
            except Exception:
                pass
            self._hide_error()

    # ══════════════════════════════════════════════════════════════════════════
    #  Forgot-password flow
    # ══════════════════════════════════════════════════════════════════════════
    def _forgot_clicked(self):
        now = time.time()
        if now < self._forgot_locked_until:
            wait = int(self._forgot_locked_until - now)
            self._show_error(
                f"Maximum amount of attempts reached. "
                f"Please wait {_fmt_time(wait)} before trying again.")
            return
        self._show_forgot_email()

    # ── Step 1: enter email ─────────────────────────────────────────────────
    def _show_forgot_email(self):
        self._clear_screen()
        self._title("Login")        # title stays as Login per spec

        prompt = ctk.CTkLabel(self.root, text="Type in your email here",
                              font=ctk.CTkFont('Segoe UI', 20, 'bold'),
                              text_color=WHITE, fg_color=BG_MAIN)
        prompt.place(relx=0.5, rely=0.40, anchor='center')
        self._add(prompt)

        self._fp_field = ctk.CTkEntry(
            self.root, placeholder_text="Email",
            width=FIELD_W, height=48, corner_radius=10,
            fg_color=BG_CARD, border_color='#2A2A40', border_width=1,
            text_color=WHITE, font=ctk.CTkFont('Segoe UI', 15))
        self._fp_field.place(relx=0.5, rely=0.46, anchor='center')
        self._add(self._fp_field)

        note = ctk.CTkLabel(
            self.root,
            text="Enter your email and a six-digit code will be sent to it.",
            font=ctk.CTkFont('Segoe UI', 12), text_color=GRAY, fg_color=BG_MAIN)
        note.place(relx=0.5, rely=0.505, anchor='center')
        self._add(note)

        send_btn = make_button(
            self.root, "Send", self._do_send_code,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=44)
        send_btn.place(relx=0.5, rely=0.58, anchor='center')
        self._add(send_btn)

        self._fp_field.bind('<Return>', lambda e: self._do_send_code())
        self._back_link()

    def _do_send_code(self):
        if self._busy:
            return
        email = self._fp_field.get().strip()
        if not email:
            self._show_error("Enter your email.")
            return
        self._reset_email = email
        # /forgot-password always returns 200 (no email enumeration).
        self._run_async('/forgot-password', {"email": email},
                        on_ok=lambda data: self._show_forgot_code(),
                        on_fail=lambda msg: self._show_error(msg))

    # ── Step 2: enter the 6-digit code ──────────────────────────────────────
    def _show_forgot_code(self):
        self._clear_screen()
        self._title("Login")

        prompt = ctk.CTkLabel(self.root, text="Enter six-digit code below",
                              font=ctk.CTkFont('Segoe UI', 20, 'bold'),
                              text_color=WHITE, fg_color=BG_MAIN)
        prompt.place(relx=0.5, rely=0.40, anchor='center')
        self._add(prompt)

        self._fp_field = ctk.CTkEntry(
            self.root, placeholder_text="6-digit code",
            width=FIELD_W, height=48, corner_radius=10,
            fg_color=BG_CARD, border_color='#2A2A40', border_width=1,
            text_color=WHITE, font=ctk.CTkFont('Segoe UI', 15),
            justify='center')
        self._fp_field.place(relx=0.5, rely=0.46, anchor='center')
        self._add(self._fp_field)

        note = ctk.CTkLabel(self.root, text="Check your email for the code.",
                            font=ctk.CTkFont('Segoe UI', 12),
                            text_color=GRAY, fg_color=BG_MAIN)
        note.place(relx=0.5, rely=0.505, anchor='center')
        self._add(note)

        verify_btn = make_button(
            self.root, "Verify", self._do_verify_code,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=40)
        verify_btn.place(relx=0.5, rely=0.58, anchor='center')
        self._add(verify_btn)

        self._fp_field.bind('<Return>', lambda e: self._do_verify_code())
        self._back_link()

    def _do_verify_code(self):
        if self._busy:
            return
        code = self._fp_field.get().strip()
        if not code:
            self._show_error("Enter the six-digit code.")
            return
        self._reset_code = code
        self._run_async('/verify-code',
                        {"email": self._reset_email, "code": code},
                        on_ok=lambda data: self._show_new_password(),
                        on_fail=self._verify_failed)

    def _verify_failed(self, msg):
        self._forgot_fails += 1
        if self._forgot_fails >= 3:
            self._forgot_fails = 0
            self._forgot_lock_count += 1
            wait = 30 * self._forgot_lock_count
            self._forgot_locked_until = time.time() + wait
            self._show_landing()
            self._show_error(
                f"Maximum amount of attempts reached. "
                f"Please wait {_fmt_time(wait)} before trying again.")
        else:
            self._show_landing()
            self._show_error("The code was incorrect. Please try again.")

    # ── Step 3: set the new password ────────────────────────────────────────
    def _show_new_password(self):
        self._clear_screen()
        self._title("Login")

        prompt = ctk.CTkLabel(self.root, text="Set your new password",
                              font=ctk.CTkFont('Segoe UI', 20, 'bold'),
                              text_color=WHITE, fg_color=BG_MAIN)
        prompt.place(relx=0.5, rely=0.40, anchor='center')
        self._add(prompt)

        self._np_field = ctk.CTkEntry(
            self.root, placeholder_text="New password (min 8 chars)", show='•',
            width=FIELD_W, height=48, corner_radius=10,
            fg_color=BG_CARD, border_color='#2A2A40', border_width=1,
            text_color=WHITE, font=ctk.CTkFont('Segoe UI', 15))
        self._np_field.place(relx=0.5, rely=0.46, anchor='center')
        self._add(self._np_field)

        save_btn = make_button(
            self.root, "Save Password", self._do_reset_password,
            base_bg=BLUE, base_fg=BLACK, hover_bg=WHITE, hover_fg=BLACK,
            width_pad=30)
        save_btn.place(relx=0.5, rely=0.55, anchor='center')
        self._add(save_btn)

        self._np_field.bind('<Return>', lambda e: self._do_reset_password())
        self._back_link()

    def _do_reset_password(self):
        if self._busy:
            return
        pw = self._np_field.get()
        if len(pw) < 8:
            self._show_error("Password must be at least 8 characters.")
            return
        self._run_async('/reset-password',
                        {"email": self._reset_email, "code": self._reset_code,
                         "new_password": pw},
                        on_ok=self._reset_done,
                        on_fail=lambda msg: self._show_error(msg))

    def _reset_done(self, data):
        self._forgot_fails = 0
        self._show_login()
        self._show_error("Password updated. You can now log in.")

    # ══════════════════════════════════════════════════════════════════════════
    #  Auth backend (uvicorn) lifecycle
    # ══════════════════════════════════════════════════════════════════════════
    def _server_reachable(self, timeout=1.0):
        """True if something already answers /health at SERVER_URL."""
        try:
            requests.get(f"{SERVER_URL}/health", timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False

    def _start_backend(self):
        """
        Launch the FastAPI auth server (uvicorn) as a child process, unless one
        is already running. The process is killed automatically when this app
        exits. Failures here never crash the GUI — the user just gets the usual
        'can't reach server' message if it doesn't come up.
        """
        # Don't start a second server if one is already listening.
        if self._server_reachable():
            return

        frozen = getattr(sys, 'frozen', False)
        if frozen:
            # The personal build places both executables in the same folder.
            server_dir = os.path.dirname(os.path.abspath(sys.executable))
            server_exe = os.path.join(
                server_dir, 'AI Music Detector Server.exe')
            if not os.path.isfile(server_exe):
                print(f"[LoginScreen] Packaged auth server not found at "
                      f"{server_exe}.")
                return
            server_command = [server_exe]
        else:
            # Source/development mode continues to launch uvicorn through the
            # active Python interpreter exactly as before.
            server_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Server")
            if not os.path.isfile(os.path.join(server_dir, "Server.py")):
                print(f"[LoginScreen] Auth server not found at {server_dir}; "
                      f"start it manually.")
                return
            server_command = [
                sys.executable, "-m", "uvicorn", "Server:app"]

        parsed = urlparse(SERVER_URL)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8000)

        # On Windows, don't pop a console window for the child process.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # Route the server's output to a log file so its messages aren't lost.
        # In dev mode (no SMTP configured) the verification code is PRINTED here
        # — read it from CodeFiles/Server/server.log while testing.
        if frozen:
            log_dir = os.path.join(
                os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
                'AI Music Detector')
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = server_dir
        log_path = os.path.join(log_dir, "server.log")
        try:
            self._server_log = open(log_path, "a", buffering=1, encoding="utf-8")
        except Exception:
            self._server_log = subprocess.DEVNULL
        try:
            if not frozen:
                server_command.extend(["--host", host, "--port", port])
            self._server_proc = subprocess.Popen(
                server_command,
                cwd=server_dir,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            # Make sure the server dies with the app.
            atexit.register(self._stop_backend)
            print(f"[LoginScreen] Started auth server (pid "
                  f"{self._server_proc.pid}) on {host}:{port}. "
                  f"Logs -> {log_path}")
        except Exception as e:
            print(f"[LoginScreen] Could not start auth server: {e}")
            self._server_proc = None

    def _stop_backend(self):
        """Terminate the auth server if we started it."""
        proc = self._server_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Networking (off the main thread) + success handoff
    # ══════════════════════════════════════════════════════════════════════════
    def _run_async(self, endpoint, payload, on_ok, on_fail):
        self._busy = True
        self._hide_error()

        def worker():
            try:
                # Retry briefly on ConnectionError — the auto-started server may
                # still be booting when the user submits.
                last_err = None
                for attempt in range(6):
                    try:
                        r = requests.post(f"{SERVER_URL}{endpoint}",
                                          json=payload, timeout=15)
                        break
                    except requests.exceptions.ConnectionError as e:
                        last_err = e
                        time.sleep(0.5)
                else:
                    raise last_err

                if r.status_code == 200:
                    data = r.json()
                    self.root.after(0, lambda: self._done(on_ok, data))
                else:
                    try:
                        raw = r.json().get('detail', 'Something went wrong.')
                        if isinstance(raw, str):
                            detail = raw
                        elif isinstance(raw, list):
                            # FastAPI/Pydantic validation error (e.g. bad email):
                            # detail is a list of dicts, not a string.
                            detail = "Please check your input and try again."
                        else:
                            detail = str(raw)
                    except Exception:
                        detail = f"Error {r.status_code}"
                    self.root.after(0, lambda: self._done(on_fail, detail))
            except requests.exceptions.ConnectionError:
                self.root.after(0, lambda: self._done(
                    on_fail, "Can't reach the server. Is it running?"))
            except Exception as e:
                self.root.after(0, lambda: self._done(on_fail, str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, callback, arg):
        self._busy = False
        callback(arg)

    def _auth_success(self, data):
        self._login_fails = 0
        self._login_lock_count = 0
        token, tier, email = data['token'], data['tier'], data['email']
        LoginLoadingScreen(self.root,
                           on_done=lambda: self._finish(token, tier, email),
                           ready_check=self._startup_ready.is_set)

    def _finish(self, token, tier, email):
        # Record the successful session, then let run() launch the next window
        # only after this callback and this Tcl event loop have fully unwound.
        # Creating a second CTk root from inside this callback left internal
        # "update" / "check_dpi_scaling" after-jobs pointing at commands that
        # disappeared when the login widgets were destroyed.
        self._handoff = (token, tier, email)
        try:
            pending = self.root.tk.call('after', 'info')
            after_ids = (pending if isinstance(pending, (tuple, list))
                         else self.root.tk.splitlist(pending))
            for after_id in after_ids:
                try:
                    self.root.after_cancel(after_id)
                except tk.TclError:
                    pass
        except tk.TclError:
            pass
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        if self._handoff is not None:
            token, tier, email = self._handoff
            self._handoff = None
            self.on_success(token, tier, email)


# ══════════════════════════════════════════════════════════════════════════════
#  Post-login loading screen (distinct from startup/main loading)
# ══════════════════════════════════════════════════════════════════════════════
class LoginLoadingScreen:
    def __init__(self, root, on_done, duration_ms=1600, ready_check=None):
        self.root = root
        self.on_done = on_done
        # Optional gate: don't hand off until this returns True (e.g. the next
        # screen has finished importing). Keeps the loading screen up instead of
        # showing a blank gap.
        self._ready_check = ready_check

        self.frame = tk.Frame(root, bg=BG_MAIN)
        self.frame.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame.lift()

        self._label = tk.Label(self.frame, text="Signing you in",
                               bg=BG_MAIN, fg=WHITE,
                               font=('Segoe UI', 28, 'bold'))
        self._label.place(relx=0.5, rely=0.46, anchor='center')

        self._bar_bg = tk.Frame(self.frame, bg='#1A1A2E', width=360, height=6)
        self._bar_bg.place(relx=0.5, rely=0.54, anchor='center')
        self._bar = tk.Frame(self._bar_bg, bg=BLUE, width=0, height=6)
        self._bar.place(x=0, y=0)

        self._dots = 0
        self._elapsed = 0
        self._duration = duration_ms
        self._animate_dots()
        self._animate_bar()

    def _animate_dots(self):
        self._dots = (self._dots + 1) % 4
        self._label.configure(text="Signing you in" + "." * self._dots)
        self._dot_job = self.root.after(300, self._animate_dots)

    def _animate_bar(self):
        self._elapsed += 30
        frac = min(1.0, self._elapsed / self._duration)
        self._bar.configure(width=int(360 * frac))
        if frac < 1.0:
            self.root.after(30, self._animate_bar)
        elif self._ready_check is not None and not self._ready_check():
            # Min time elapsed but the next screen isn't ready yet — hold here
            # (bar full, dots still animating) instead of blanking out.
            self.root.after(50, self._animate_bar)
        else:
            try:
                self.root.after_cancel(self._dot_job)
            except Exception:
                pass
            self.frame.destroy()
            self.on_done()


# ══════════════════════════════════════════════════════════════════════════════
#  Launch: show login, then on success launch the Upload Track (startup) screen.
# ══════════════════════════════════════════════════════════════════════════════
def launch():
    def after_login(token, tier, email):
        print(f"[login] success: {email}  tier={tier}")
        import appGUI_Startup
        appGUI_Startup.CURRENT_TIER = tier      # override the hardcoded default
        appGUI_Startup.CURRENT_USER_EMAIL = email
        appGUI_Startup.AUTH_TOKEN = token
        startup_app = appGUI_Startup.AIDetectorApp()
        startup_app.run()

    LoginScreen(on_success=after_login).run()


if __name__ == '__main__':
    launch()
